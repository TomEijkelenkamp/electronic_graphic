import torch
import torchvision.utils
import os
from bezier import draw_bezier_curve
from datetime import datetime


def run(model, dataloader_train, dataloader_validate, optimizer, scheduler, num_epochs=500, start_epoch=0, num_curves=8, save_and_evaluate_every=500, model_path=os.path.join("models", "checkpoint.pth"), image_folder=os.path.join("results", "images")):
    """Runs the training and validation loop."""
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch}/{num_epochs}, Learning Rate: {scheduler.get_last_lr()}")
        
        train(model, dataloader_train, dataloader_validate, optimizer, scheduler, num_curves=num_curves, epoch=epoch, save_and_evaluate_every=save_and_evaluate_every, model_path=model_path, image_folder=image_folder)  # Train the model

        if model.check_for_invalid_params():
            print("Invalid parameters detected. Stopping training.")
            break

        model.save_checkpoint(optimizer, scheduler, epoch, model_path=model_path)  # Save model checkpoint

        validate(model, dataloader_validate, num_curves=num_curves, image_folder=image_folder)



def train(model, dataloader_train, dataloader_validate, optimizer, scheduler, num_curves=8, epoch=0, save_and_evaluate_every=500, model_path=os.path.join("models", "checkpoint.pth"), image_folder=os.path.join("results", "images")):
    """Trains the model on a given dataset."""
    device = model.device  # Get the device from the model

    print("Training...")

    start_time = datetime.now()

    for batch_idx, batch in enumerate(dataloader_train):
        # Move batch to the appropriate device
        example_batch = batch.to(device)

        batch_size, _, height, width = example_batch.shape

        # Shape: [batch_size, 1, height, width]
        canvas = torch.ones(batch_size, 3, height, width, device=device, requires_grad=True)

        for i in range(num_curves):

            optimizer.zero_grad()  # Reset gradients

            # Control points in the range [0, 1]
            control_points, color, thickness, sharpness = model(example_batch, canvas)  # Shape: [batch_size, num_points, 2]
            
            # Draw the curves
            new_canvas = draw_bezier_curve(control_points, color, thickness, sharpness, canvas)

            # Compute the mse loss between the new canvas and the example batch
            loss = torch.nn.functional.mse_loss(new_canvas, example_batch)
            
            loss.backward()  # Retain graph for further backward passes
            optimizer.step()  # Update weights

            canvas = new_canvas.detach().clone()  # Detach the canvas to avoid tracking gradients
            
        if batch_idx % 50 == 0 and batch_idx > 0:  # Print every 50 batches
            print(f"Batch {batch_idx}, Loss: {loss.item()}, Time: {datetime.now() - start_time}")
            start_time = datetime.now()  # Reset start time for next batch
            if model.check_for_invalid_params():
                break

        if batch_idx % save_and_evaluate_every == 0 and batch_idx > 0:  # Limit the number of batches for training
            if model.check_for_invalid_params():
                break

            model.save_checkpoint(optimizer, scheduler, epoch, model_path=model_path)  # Save model checkpoint

            validate(model, dataloader_validate, num_curves=num_curves, image_folder=image_folder)

    scheduler.step()  # Update learning rate


def validate(model, dataloader, num_curves=8, image_folder=os.path.join("results", "images")):
    """Validates the model on a given dataset."""
    device = model.device  # Get the device from the model

    print("Validating...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > 10:
                break

            example_batch = batch.to(device)

            batch_size, _, height, width = example_batch.shape

            # Shape: [batch_size, 1, height, width]
            canvas = torch.ones(batch_size, 3, height, width, device=device)
            fullsize_canvas = torch.ones(batch_size, 3, 512, 512, device=device)

            for i in range(num_curves):
                # Control points in the range [0, 1]
                control_points, color, thickness, sharpness = model(example_batch, canvas)  # Shape: [batch_size, num_points, 2]

                # Draw the curves
                new_canvas = draw_bezier_curve(control_points, color, thickness, sharpness, canvas)

                # Detach the canvas to avoid tracking gradients
                canvas = new_canvas.detach().clone()

                fullsize_canvas = draw_bezier_curve(control_points, color, thickness*8.0, sharpness*8.0, fullsize_canvas)

            # Save the canvas to a file
            save_results(canvas, example_batch, fullsize_canvas, prefix=f"valid_{batch_idx}", image_folder=image_folder)

        print(f"Validation completed. Results saved to '{image_folder}'")

def save_results(canvas, example, fullsize_canvas, prefix="output", image_folder=os.path.join("results", "images")):
    """Saves each canvas in the batch as a separate PNG image."""
    if not os.path.exists(image_folder):
        os.makedirs(image_folder, exist_ok=True)

    for i in range(canvas.shape[0]):
        # Concat the canvas with the original image for visualization
        upscaled_example = torch.nn.functional.interpolate(
            example[i].unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False
        ).squeeze(0)
        combined = torch.cat((upscaled_example, fullsize_canvas[i]), dim=2)

        filename = os.path.join(image_folder, f"{prefix}_{i}.png")

        torchvision.utils.save_image(combined, filename)



            


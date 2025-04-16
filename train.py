import torch
import torchvision.utils
import os
from bezier import draw_bezier_curve
from tqdm import tqdm
from collections import deque
from dataset import get_train_dataset_loader, get_val_dataset_loader

def train(model, dataset, image_width, image_height, batch_size_train, batch_size_val, num_val_images, optimizer, scheduler, num_epochs=500, start_epoch=0, num_curves=8, save_and_evaluate_every=500, model_path=os.path.join("models", "checkpoint.pth"), image_folder=os.path.join("results", "images")):
    """Runs the training and validation loop."""
    train_loader = get_train_dataset_loader(dataset, image_width=image_width, image_height=image_height, batch_size=batch_size_train)

    device = model.device  # Get the device from the model

    loss_history = deque(maxlen=50*num_curves)  # Automatically keeps only the last 50 losses

    print("Training...")
    for epoch in range(start_epoch, num_epochs):
        
        lr = scheduler.get_last_lr()[0]
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}, Learning rate: {lr:.5f}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(progress):
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

                loss_history.append(loss.item())
                
                loss.backward()  # Retain graph for further backward passes
                optimizer.step()  # Update weights

                canvas = new_canvas.detach().clone()  # Detach the canvas to avoid tracking gradients
                
            should_log = batch_idx % 10 == 0 and batch_idx > 0
            should_save = batch_idx % save_and_evaluate_every == 0 and (batch_idx > 0 or epoch > 0)

            if should_log or should_save:
                avg_loss = sum(loss_history) / len(loss_history)
                progress.set_postfix(loss=loss.item(), avg_loss=f"{avg_loss:.4f}")

                if model.check_for_invalid_params():
                    tqdm.write("\033[91m🔴 ERROR:\033[0m Invalid parameters detected. Stopping training.")
                    break

                if should_save:
                    model.save_checkpoint(optimizer, scheduler, epoch, avg_loss, model_path=model_path)
                    validate(model, dataset, image_width, image_height, batch_size_val, num_val_images, num_curves=num_curves, image_folder=image_folder)
          
        scheduler.step()  # Update learning rate

def validate(model, dataset, image_width, image_height, batch_size_val, num_val_images, num_curves=8, image_folder=os.path.join("results", "images")):
    """Validates the model on a given dataset."""

    val_loader = get_val_dataset_loader(dataset, num_images=num_val_images, image_width=image_width, image_height=image_height, batch_size=batch_size_val)

    device = model.device  # Get the device from the model

    GREEN = "\033[38;5;34m"
    RESET = "\033[0m"

    with torch.no_grad():
        progress = tqdm(val_loader, desc=f"{GREEN}Generating validation images to {image_folder}{RESET}", leave=True)

        for batch_idx, batch in enumerate(progress):

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



            


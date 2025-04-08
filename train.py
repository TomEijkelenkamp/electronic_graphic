import torch
import torchvision.utils
import os
from bezier import draw_bezier_curve


def train(model, dataloader, optimizer, scheduler, num_epochs=500, start_epoch=0, num_curves=8, model_path=os.path.join("models", "checkpoint.pth"), image_folder=os.path.join("results", "images")):

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    model.to(device)  # Move model to the appropriate device

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        
        for batch_idx, batch in enumerate(dataloader):
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
                
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch}, Learning Rate: {scheduler.get_last_lr()}")

        # Save the canvas to a file
        save_results(new_canvas, example_batch, prefix=f"mnist_{epoch}", image_folder=image_folder)

        model.save_checkpoint(optimizer, scheduler, epoch, model_path=model_path)  # Save model checkpoint


def save_results(canvas, example, prefix="output", image_folder=os.path.join("results", "images")):
    """Saves each canvas in the batch as a separate PNG image."""
    if not os.path.exists(image_folder):
        os.makedirs(image_folder, exist_ok=True)
    batch_size = canvas.shape[0]
    for i in range(batch_size):
        # Concat the canvas with the original image for visualization
        combined = torch.cat((example[i], canvas[i]), dim=2)

        filename = os.path.join(image_folder, f"{prefix}_{i}.png")

        torchvision.utils.save_image(combined, filename)



            


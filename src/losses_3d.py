import torch

def gradient_penalty_3d(critic, real, fake, device="cpu", add_grad = False):
    BATCH_SIZE, C, D, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1, 1)).repeat(1, C, D, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    if add_grad:
        interpolated_images.requires_grad=True
    
    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

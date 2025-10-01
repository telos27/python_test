"""
Subtle PyTorch Training Bugs

This file demonstrates nuanced training scenarios that are challenging to detect:
- Indirect state mutations through helper functions
- Multi-step control flow with state dependencies
- Cross-module state interactions
- Temporal ordering violations
- Advanced gradient management patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Callable, Optional


def prepare_model_for_evaluation(model):
    """Prepare model for evaluation phase"""
    model.eval()
    return model


def execute_training_iteration(model, data, target, optimizer):
    """Execute a single training iteration"""
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def mixed_mode_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    prepare_model_for_evaluation(model)
    loss = execute_training_iteration(model, data, target, optimizer)


def process_batch(model, data, target, optimizer, apply_updates):
    """Process a single batch with optional gradient computation"""
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    if apply_updates:
        optimizer.zero_grad()
        loss.backward()

    return loss


def adaptive_training_step(model, data, target, optimizer, use_gradients):
    """Adaptive training based on runtime conditions"""
    loss = process_batch(model, data, target, optimizer, use_gradients)
    optimizer.step()


def conditional_training_loop():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for i in range(10):
        data = torch.randn(32, 10)
        target = torch.randint(0, 2, (32,))
        adaptive_training_step(model, data, target, optimizer, use_gradients=(i % 2 == 0))


def compute_ensemble_loss(auxiliary_model, data, target):
    """Compute loss from auxiliary model"""
    output = auxiliary_model(data)
    loss = nn.functional.cross_entropy(output, target)
    return loss


def train_primary_with_ensemble(primary_model, auxiliary_model, optimizer, data, target):
    """Train primary model using ensemble feedback"""
    optimizer.zero_grad()
    loss = compute_ensemble_loss(auxiliary_model, data, target)
    loss.backward()
    optimizer.step()


def ensemble_training():
    primary_model = nn.Linear(10, 2)
    auxiliary_model = nn.Linear(10, 2)
    optimizer = optim.SGD(primary_model.parameters(), lr=0.01)
    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    train_primary_with_ensemble(primary_model, auxiliary_model, optimizer, data, target)


def resilient_training_step(model, data, target, optimizer):
    """Training step with error resilience"""
    optimizer.zero_grad()

    try:
        output = model(data)

        if torch.rand(1).item() > 0.5:
            raise ValueError("Unstable training condition detected")

        loss = nn.functional.cross_entropy(output, target)
        loss.backward()

    except ValueError:
        print("Recovering from training instability...")

    finally:
        optimizer.step()


def robust_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for _ in range(5):
        data = torch.randn(32, 10)
        target = torch.randint(0, 2, (32,))
        resilient_training_step(model, data, target, optimizer)


def accumulate_with_adaptive_scaling(model, dataloader, optimizer, accumulation_steps):
    """Gradient accumulation with adaptive loss scaling"""
    optimizer.zero_grad()

    for i, (data, target) in enumerate(dataloader):
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)

        if i % 2 == 0:
            loss = loss / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


def advanced_accumulation():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dataloader = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(8)]

    accumulate_with_adaptive_scaling(model, dataloader, optimizer, accumulation_steps=4)


def hierarchical_training(model, data, target, optimizer, level):
    """Recursive training pattern with depth-based logic"""
    if level > 0:
        model.eval()
        hierarchical_training(model, data, target, optimizer, level - 1)

    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    if level == 0:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def multi_level_training():
    model = nn.Linear(10, 2)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    hierarchical_training(model, data, target, optimizer, level=2)


def create_update_handler(optimizer):
    """Factory for parameter update handlers"""
    def handle_update():
        optimizer.step()
        optimizer.zero_grad()
    return handle_update


def callback_driven_training(model, data, target, optimizer):
    """Training with callback pattern"""
    update_handler = create_update_handler(optimizer)

    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    optimizer.zero_grad()
    loss.backward()

    update_handler()
    optimizer.step()


def callback_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    callback_driven_training(model, data, target, optimizer)


def perform_optimization(primary_opt, secondary_opt, data, target, model):
    """Perform optimization using two optimizer references"""
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    primary_opt.zero_grad()
    loss.backward()

    secondary_opt.step()
    primary_opt.step()


def train_with_multiple_references(model, data, target):
    """Training using multiple optimizer references"""
    main_optimizer = optim.SGD(model.parameters(), lr=0.01)
    backup_optimizer = main_optimizer

    perform_optimization(main_optimizer, backup_optimizer, data, target, model)


def multi_reference_training():
    model = nn.Linear(10, 2)
    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    train_with_multiple_references(model, data, target)


class TrainingContext:
    """Shared training context across operations"""
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)
        self.loss = None

    def compute_gradients(self, data, target):
        """Compute gradients for given batch"""
        output = self.model(data)
        self.loss = nn.functional.cross_entropy(output, target)
        self.optimizer.zero_grad()
        self.loss.backward()

    def apply_updates(self):
        """Apply parameter updates"""
        self.optimizer.step()


def shared_context_training():
    model = nn.Linear(10, 2)
    ctx = TrainingContext(model)

    data1 = torch.randn(32, 10)
    target1 = torch.randint(0, 2, (32,))
    data2 = torch.randn(32, 10)
    target2 = torch.randint(0, 2, (32,))

    ctx.compute_gradients(data1, target1)
    ctx.compute_gradients(data2, target2)
    ctx.apply_updates()


class ModelController:
    """Controls model configuration"""
    def __init__(self, model):
        self.model = model

    def configure_for_validation(self):
        """Configure model for validation"""
        self.model.eval()


class OptimizerController:
    """Controls optimization process"""
    def __init__(self, model):
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)

    def perform_update(self, model, data, target):
        """Perform optimization step"""
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def modular_training():
    model = nn.Linear(10, 2)

    model_ctrl = ModelController(model)
    optimizer_ctrl = OptimizerController(model)

    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    model_ctrl.configure_for_validation()
    optimizer_ctrl.perform_update(model, data, target)


def configure_model_mode(model, is_training):
    """Configure model training/eval mode"""
    if is_training:
        model.train()
    else:
        model.eval()


def train_with_mode_switching(model, train_data, val_data, train_target, val_target, optimizer):
    """Training with mode switching"""
    configure_model_mode(model, is_training=False)

    val_output = model(val_data)
    val_loss = nn.functional.cross_entropy(val_output, val_target)

    train_output = model(train_data)
    train_loss = nn.functional.cross_entropy(train_output, train_target)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


def mode_switching_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_data = torch.randn(32, 10)
    train_target = torch.randint(0, 2, (32,))
    val_data = torch.randn(32, 10)
    val_target = torch.randint(0, 2, (32,))

    train_with_mode_switching(model, train_data, val_data, train_target, val_target, optimizer)


def compute_auxiliary_loss(model, data):
    """Compute auxiliary loss for regularization"""
    activations = model(data)
    return activations.pow(2).mean()


def train_with_detached_auxiliary(model, data, target, optimizer):
    """Training with detached auxiliary loss"""
    output = model(data)
    main_loss = nn.functional.cross_entropy(output, target)

    aux_loss = compute_auxiliary_loss(model, data)
    aux_loss = aux_loss.detach()

    total_loss = main_loss + aux_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


def detached_loss_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    train_with_detached_auxiliary(model, data, target, optimizer)


class GradientAccumulator:
    """Manages gradient accumulation state"""
    def __init__(self, accumulation_steps):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_update(self):
        """Check if we should update parameters"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0

    def reset(self):
        """Reset accumulation counter"""
        self.current_step = 0


def train_with_accumulator(model, dataloader, optimizer, accumulator):
    """Training with gradient accumulator"""
    for data, target in dataloader:
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)

        loss.backward()

        if accumulator.should_update():
            optimizer.step()
            optimizer.zero_grad()


def accumulator_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    accumulator = GradientAccumulator(accumulation_steps=4)

    dataloader = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(8)]

    train_with_accumulator(model, dataloader, optimizer, accumulator)


def apply_weight_norm(model):
    """Apply weight normalization to model"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            torch.nn.utils.weight_norm(module)


def train_with_weight_norm(model, data, target, optimizer):
    """Training with weight normalization"""
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    optimizer.zero_grad()
    loss.backward()

    apply_weight_norm(model)

    optimizer.step()


def weight_norm_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    train_with_weight_norm(model, data, target, optimizer)


def freeze_backbone(model):
    """Freeze early layers of the model"""
    for param in model.features.parameters():
        param.requires_grad = False


def train_with_frozen_layers(model, data, target, optimizer):
    """Train model with some frozen layers"""
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def frozen_layer_training():
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Linear(10, 5)
            self.classifier = nn.Linear(5, 2)

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    freeze_backbone(model)

    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))
    train_with_frozen_layers(model, data, target, optimizer)


def apply_gradient_clipping(model, max_norm):
    """Apply gradient clipping to model parameters"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def train_with_clipping(model, data, target, optimizer):
    """Training with gradient clipping"""
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    apply_gradient_clipping(model, max_norm=1.0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def gradient_clipping_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    train_with_clipping(model, data, target, optimizer)


class CustomScheduler:
    """Custom learning rate scheduler"""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0

    def step(self):
        """Update learning rate"""
        self.step_count += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.95


def train_with_scheduler(model, data, target, optimizer, scheduler):
    """Training with learning rate scheduling"""
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    scheduler.step()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def scheduler_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = CustomScheduler(optimizer)

    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    train_with_scheduler(model, data, target, optimizer, scheduler)


def compute_loss_with_regularization(model, data, target, lambda_reg):
    """Compute loss with L2 regularization"""
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    l2_reg = sum(p.pow(2).sum() for p in model.parameters())
    loss = loss + lambda_reg * l2_reg

    return loss


def train_with_manual_regularization(model, data, target, optimizer):
    """Train with manual regularization"""
    loss = compute_loss_with_regularization(model, data, target, lambda_reg=0.01)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def double_regularization_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    train_with_manual_regularization(model, data, target, optimizer)


def accumulate_losses(losses: List[torch.Tensor]) -> torch.Tensor:
    """Accumulate multiple losses"""
    total_loss = sum(losses)
    return total_loss


def multi_task_training(model, data, targets, optimizer):
    """Multi-task learning with loss accumulation"""
    optimizer.zero_grad()

    losses = []
    for target in targets:
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        losses.append(loss)

    total_loss = accumulate_losses(losses)

    optimizer.step()


def multi_task_example():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data = torch.randn(32, 10)
    targets = [torch.randint(0, 2, (32,)) for _ in range(3)]

    multi_task_training(model, data, targets, optimizer)


def save_checkpoint(model, optimizer, path):
    """Save model and optimizer checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path):
    """Load model and optimizer checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_with_checkpointing(model, data, target, optimizer, checkpoint_path):
    """Training with checkpoint management"""
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    optimizer.zero_grad()
    loss.backward()

    load_checkpoint(model, optimizer, checkpoint_path)

    optimizer.step()


def warm_restart_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    checkpoint_path = "/tmp/checkpoint.pt"
    save_checkpoint(model, optimizer, checkpoint_path)

    train_with_checkpointing(model, data, target, optimizer, checkpoint_path)


class MomentumTracker:
    """Track and manipulate optimizer momentum"""
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def reset_momentum(self):
        """Reset momentum buffers"""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()


def train_with_momentum_reset(model, data, target, optimizer, momentum_tracker):
    """Training with momentum management"""
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    optimizer.zero_grad()
    loss.backward()

    momentum_tracker.reset_momentum()

    optimizer.step()


def momentum_training():
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    tracker = MomentumTracker(optimizer)

    data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))

    train_with_momentum_reset(model, data, target, optimizer, tracker)


def detach_hidden_state(hidden):
    """Detach hidden state for truncated backprop"""
    return hidden.detach()


def train_sequence_model(model, sequences, optimizer):
    """Train RNN with truncated backprop through time"""
    optimizer.zero_grad()

    hidden = None
    for seq in sequences:
        output, hidden = model(seq, hidden)
        loss = output.sum()
        loss.backward()

        hidden = detach_hidden_state(hidden)

    optimizer.step()


def truncated_bptt_training():
    class SimpleRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(10, 20, batch_first=True)

        def forward(self, x, hidden=None):
            output, hidden = self.rnn(x, hidden)
            return output, hidden

    model = SimpleRNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    sequences = [torch.randn(32, 5, 10) for _ in range(4)]

    train_sequence_model(model, sequences, optimizer)


def mixed_precision_step(model, data, target, optimizer, scaler):
    """Training step with mixed precision"""
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def mixed_precision_training():
    if not torch.cuda.is_available():
        return

    model = nn.Linear(10, 2).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler()

    data = torch.randn(32, 10).cuda()
    target = torch.randint(0, 2, (32,)).cuda()

    mixed_precision_step(model, data, target, optimizer, scaler)


if __name__ == "__main__":
    print("=" * 80)
    print("Subtle PyTorch Training Bugs")
    print("=" * 80)

    examples = [
        ("Mixed Mode Training", mixed_mode_training),
        ("Conditional Training Loop", conditional_training_loop),
        ("Ensemble Training", ensemble_training),
        ("Robust Training", robust_training),
        ("Advanced Accumulation", advanced_accumulation),
        ("Multi-Level Training", multi_level_training),
        ("Callback Training", callback_training),
        ("Multi-Reference Training", multi_reference_training),
        ("Shared Context Training", shared_context_training),
        ("Modular Training", modular_training),
        ("Mode Switching Training", mode_switching_training),
        ("Detached Loss Training", detached_loss_training),
        ("Accumulator Training", accumulator_training),
        ("Weight Norm Training", weight_norm_training),
        ("Frozen Layer Training", frozen_layer_training),
        ("Gradient Clipping Training", gradient_clipping_training),
        ("Scheduler Training", scheduler_training),
        ("Double Regularization Training", double_regularization_training),
        ("Multi-Task Example", multi_task_example),
        ("Warm Restart Training", warm_restart_training),
        ("Momentum Training", momentum_training),
        ("Truncated BPTT Training", truncated_bptt_training),
        ("Mixed Precision Training", mixed_precision_training),
    ]

    for name, func in examples:
        print(f"\n{name}")
        print("-" * 80)
        try:
            func()
            print("✓ Completed")
        except Exception as e:
            print(f"✗ Error: {e}")

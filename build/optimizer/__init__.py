from models.optimizers.optimizer_with_scheduler import OptimizerWithScheduler
from utils.register import register


def optim_instance(model, config):
    optim_config = config['optim']
    optim_config['args']['params'] = model.parameters()
    optimizer = register.get_instance_by_name(optim_config['name'], optim_config['args'], 'optim')

    scheduler_config = config['scheduler']
    base_lr = [group['lr'] for group in optimizer.param_groups]
    scheduler_config['args']['base_lr'] = base_lr
    scheduler = register.get_instance_by_name(scheduler_config['name'], scheduler_config['args'], 'scheduler')
    optimizer_with_scheduler = OptimizerWithScheduler(optimizer=optimizer, scheduler=scheduler)
    return optimizer_with_scheduler

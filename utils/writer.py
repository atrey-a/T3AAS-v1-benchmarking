from torch.utils.tensorboard import SummaryWriter

def create_tb_log(log_dir):
    return SummaryWriter(log_dir=log_dir)

def log_scalar(writer, tag, value, step):
    writer.add_scalar(tag, value, step)

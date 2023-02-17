# train
epoch = 1000
num_classes = 2
batch_size = 10
device = 'cuda:0'  # cpu or 'cuda:0'
train_image_path = 'data/Train/Healthy'  # One folder per category, categories use numbers
valid_image_path = 'data/Valid/Healthy'  # One folder per category, categories use numbers
num_workers = 0  # Loading data set thread concurrency
best_loss = 0.001  # When the loss is less than or equal to this value, the model will be saved
save_model_iter = 500  # how many times to save a model
model_output_dir = 'runs'
resume = False  # Whether to start training from the breakpoint
chkpt = ''  # Model trained with breakpoints
lr = 0.0001

# predict
predict_model = ''
predict_image_path = ''  # One folder per category, categories use numbers


image_format = 'jpg'

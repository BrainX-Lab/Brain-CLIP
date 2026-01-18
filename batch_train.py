from train import *
from model import Brain_CLIP
import matplotlib.pyplot as plt
import copy

MODEL_NAME = "H_paper"
TRIALS = 10
MAX_EPOCHS = 1000

N_ROI = 148
BATCH_SIZE = 64

device = 'cuda'
torch.set_default_device(device)
print("device:", torch.get_default_device())

# data
SC_data, FC_data = load_data(device=device)
SC_data, FC_data = preprocess_data(SC_data, FC_data)
train_dataset, val_dataset = split_data(SC_data, FC_data, ratio=0.2, device=device)

# Default values are what are used in the paper
    # k: exponent on Truth Cosine Similarity
    # alpha: trade-off between cross-modality vs. same modality similarity losses
    # beta: trade-off between Brain-CLIP vs. regeneration/prediction losses
def train_model(n_epochs=1000, k=5.0, alpha=1.0, beta=0.2, lr=10**-4, layer_sizes=[128,64,32], patience=None):
    print("k:\t\t", k)
    print("alpha:\t\t", alpha)
    print("beta:\t\t", beta)
    print("lr:\t\t", lr)
    print("layer_sizes:\t", layer_sizes)
    
    n_roi = N_ROI
    model = Brain_CLIP(dim_input=n_roi, 
                       num_heads=4, 
                       encoder_layer_sizes=layer_sizes,
                       decoder_layer_sizes=layer_sizes[::-1])

    # training parameters
    batch_size = BATCH_SIZE
    encoder_loss_fn = selfsim_loss
    decoder_loss_fn = custom_decoder_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs_without_improvement = 0
    best_version = None
    min_loss_v = None
    training_losses = []
    for epoch in range(n_epochs):
        print("Epoch:", epoch)
        
        # train
        loss = train(model, batch_size=min(len(train_dataset), batch_size), 
                    dataset=train_dataset, 
                    optimizer=optimizer, 
                    encoder_loss_fn=encoder_loss_fn,
                    decoder_loss_fn=decoder_loss_fn,
                    k=k, alpha=alpha, beta=beta,
                    device=device
                    )
        # scheduler.step()
        
        if (patience and patience > 0):
            loss_v = test(model, batch_size=min(len(val_dataset), batch_size),
                          dataset=val_dataset,
                          encoder_loss_fn=encoder_loss_fn,
                          decoder_loss_fn=decoder_loss_fn,
                          k=k, alpha=alpha,
                          device=device
                          )
            temp = loss_v[0] + beta*loss_v[1]
            
            if not min_loss_v:
                min_loss_v = temp
                best_version = copy.deepcopy(model.state_dict())
            elif temp < min_loss_v: 
                epochs_without_improvement = 0
                min_loss_v = temp
                best_version = copy.deepcopy(model.state_dict())
            else: epochs_without_improvement += 1
            
            print("\t", loss_v[0], loss_v[1], epochs_without_improvement)
            if epochs_without_improvement >= patience: break
                   
        # print losses every 10 epochs
        if (epoch % 10) == 0:
            training_losses.append(loss)

    model.load_state_dict(best_version)
    return model

for k in range(TRIALS):      
    print("-----------------------TRAINING NEW MODEL-----------------------")
    print("model:\t", MODEL_NAME)
    print("trial:\t", k)
    
    model = train_model(n_epochs=MAX_EPOCHS, patience=50)
    MODEL_SAVE_PATH = os.path.join(PATH, "Saved_Models", "t_batch_HCP", MODEL_NAME, "Model_{}_trial{}".format(MODEL_NAME, k))
    torch.save(model, MODEL_SAVE_PATH)
    
    print("\n\n\n")
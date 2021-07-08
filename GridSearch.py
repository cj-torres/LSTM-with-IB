import GaussianLSTM as gl
import torch
import csv
MODEL_DIR = "models"

hidden_size = 30
num_layers = 1
embedding_dim = 30
betas = [.001]
candidate_va = [True,False]
trials = [1,2,3,4,5,6,7,8]


with open('va_models_mi.csv', 'w', newline='') as model_file:
    model_writer = csv.writer(model_file)
    model_writer.writerow(["name","embedding_dim","annealer","loss","ce","hib","cib"])

    #for hidden_size in candidate_h_size:
        #for num_layers in candidate_num_layers:
    for va in candidate_va:
        for trial in trials:
            for beta in betas:
                avg_loss, avg_ce_loss, avg_hib_loss, avg_cib_loss, model, vocab = gl.train_unimorph_recursive_lm("corpus/eng", hidden_size, num_layers, 16384,
                                                                                                           num_epochs=200, print_every=2, noise="both",annealing="none",embedding_dim=embedding_dim,
                                                                                                                 va=va,beta_end=beta)
                name = "va_char_small_model_trial%s_va%s_beta%s" % (trial, va, str(beta)[1:])
                torch.save(model, (name+".pt"))
                model_writer.writerow([name,embedding_dim,va,avg_loss.item(),
                                       avg_ce_loss.item(),avg_hib_loss.item(),avg_cib_loss.item()])
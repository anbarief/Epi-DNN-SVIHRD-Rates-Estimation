import random
bernoulli = random.binomialvariate #name the random generator as Bernoulli
import torch
import torch.nn as nn
import torch.optim as optim
import pandas 
import numpy as np
import matplotlib.pyplot as plt


index_start = 2*60 - 2*7 #1 April 2020
index_end = 2*640 - 2*7  #2 Nov 2021 
N_p = 10.328*(10**6)     #Initial population of Sweden at t=0 (2 February 2020), using estimates of 1 January 2020 population (WPP UN data)
t_data = np.array(list(pandas.read_csv('t_ready_SVIHRD.csv')['t'][index_start:index_end+1]))
n = len(t_data)
dt = 1/2
Dt = torch.tensor(dt) 

torch.manual_seed(1);

loss_balancing = True

#Setup and Normalize Epidemic Data
S_data = np.array(list(pandas.read_csv('S_ready_SVIHRD.csv')['S'][index_start:index_end+1]))/N_p
V_data = np.array(list(pandas.read_csv('V_ready_SVIHRD.csv')['V'][index_start:index_end+1]))/N_p
I_data = np.array(list(pandas.read_csv('I_ready_SVIHRD.csv')['I'][index_start:index_end+1]))/N_p
H_data = np.array(list(pandas.read_csv('H_ready_SVIHRD.csv')['H'][index_start:index_end+1]))/N_p
R_data = np.array(list(pandas.read_csv('R_ready_SVIHRD.csv')['R'][index_start:index_end+1]))/N_p
D_data = np.array(list(pandas.read_csv('D_ready_SVIHRD.csv')['D'][index_start:index_end+1]))/N_p

diff_S = (S_data[1:]-S_data[0:-1])/dt
diff_V = (V_data[1:]-V_data[0:-1])/dt
diff_I = (I_data[1:]-I_data[0:-1])/dt
diff_H = (H_data[1:]-H_data[0:-1])/dt
diff_R = (R_data[1:]-R_data[0:-1])/dt
diff_D = (D_data[1:]-D_data[0:-1])/dt

mean_t = np.mean(t_data); std_t = np.std(t_data)
mean_S = np.mean(S_data); std_S = np.std(S_data)
mean_V = np.mean(V_data); std_V = np.std(V_data)
mean_I = np.mean(I_data); std_I = np.std(I_data)
mean_H = np.mean(H_data); std_H = np.std(H_data)
mean_R = np.mean(R_data); std_R = np.std(R_data)
mean_D = np.mean(D_data); std_D = np.std(D_data)

mean_S_diff = np.mean(diff_S); std_S_diff = np.std(diff_S)
mean_V_diff = np.mean(diff_V); std_V_diff = np.std(diff_V)
mean_I_diff = np.mean(diff_I); std_I_diff = np.std(diff_I)
mean_H_diff = np.mean(diff_H); std_H_diff = np.std(diff_H)
mean_R_diff = np.mean(diff_R); std_R_diff = np.std(diff_R)
mean_D_diff = np.mean(diff_D); std_D_diff = np.std(diff_D)

t_tensor = (t_data-mean_t)/std_t
S_tensor = (S_data-mean_S)/std_S
V_tensor = (V_data-mean_V)/std_V
I_tensor = (I_data-mean_I)/std_I
H_tensor = (H_data-mean_H)/std_H
R_tensor = (R_data-mean_R)/std_R
D_tensor = (D_data-mean_D)/std_D

diff_S_tensor = (diff_S-mean_S_diff)/std_S_diff
diff_V_tensor = (diff_V-mean_V_diff)/std_V_diff
diff_I_tensor = (diff_I-mean_I_diff)/std_I_diff
diff_H_tensor = (diff_H-mean_H_diff)/std_H_diff
diff_R_tensor = (diff_R-mean_R_diff)/std_R_diff
diff_D_tensor = (diff_D-mean_D_diff)/std_D_diff
### Ends here


### Setup Data to Feed the Input Layer of NN 
input_tensor = torch.tensor(np.array([[S_tensor[i], V_tensor[i], I_tensor[i], H_tensor[i], R_tensor[i], D_tensor[i], \
                                       diff_S_tensor[i], diff_V_tensor[i], diff_I_tensor[i], diff_H_tensor[i], diff_R_tensor[i], diff_D_tensor[i]] for i in range(n-1)], dtype=np.float32))
### Ends here


S_data = torch.tensor(S_data, dtype=torch.float32).reshape(-1, 1)
V_data = torch.tensor(V_data, dtype=torch.float32).reshape(-1, 1)
I_data = torch.tensor(I_data, dtype=torch.float32).reshape(-1, 1)
H_data = torch.tensor(H_data, dtype=torch.float32).reshape(-1, 1)
R_data = torch.tensor(R_data, dtype=torch.float32).reshape(-1, 1)
D_data = torch.tensor(D_data, dtype=torch.float32).reshape(-1, 1)


## Neural Network with ResNet
class ParameterModel(nn.Module):
    def __init__(self):
        super(ParameterModel, self).__init__()
        self.fc1 = nn.Linear(12, 28)
        self.fc2 = nn.Linear(28, 28)
        self.fc3 = nn.Linear(28, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.tanh(self.fc1(x))
        x2 = self.tanh(self.fc2(x1)); 
        x = self.sigmoid(self.fc3(x2))
        return x
## Ends here

## Instantiate the seven surrogate neural networks
beta_model = ParameterModel()
eta_model = ParameterModel() 
gammai_model = ParameterModel()
gammah_model = ParameterModel()
deltai_model = ParameterModel()
deltah_model = ParameterModel()
vrate_model = ParameterModel()
## Ends here



## Runge-Kutta (RK4) Implementation Functions
def f_S(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate):
    return (-beta*S*(I) - vrate*S - eta*S*(I))

def f_V(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate):
    return (vrate*S)

def f_I(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate):
    return  (beta*S*(I) - gammai*I -deltai*I )

def f_H(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate):
    return (eta*S*I - gammah*H - deltah*H)

def f_R(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate):
    return (gammai*I + gammah*H)

def f_D(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate):
    return (deltai*I + deltah*H)

def F(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate,dt):
    return dt*f_S(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate),\
           dt*f_V(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate),\
           dt*f_I(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate),\
           dt*f_H(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate),\
           dt*f_R(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate),\
           dt*f_D(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate)
    
def RK4(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate,dt):
    K1 = F(S,V,I,H,R,D,beta,eta,gammai,gammah,deltai,deltah,vrate,dt)
    K2 = F(S + K1[0]/2, V + K1[1]/2, I + K1[2]/2, H + K1[3]/2, R + K1[4]/2, D + K1[5]/2, beta, eta, gammai, gammah, deltai, deltah, vrate,dt)
    K3 = F(S + K2[0]/2, V + K2[1]/2, I + K2[2]/2, H + K2[3]/2, R + K2[4]/2, D + K2[5]/2, beta, eta, gammai, gammah, deltai, deltah, vrate,dt)
    K4 = F(S + K3[0], V + K3[1], I + K3[2], H + K3[3], R + K3[4], D + K3[5], beta, eta, gammai, gammah, deltai, deltah, vrate,dt)
    return S + (1/6)*(K1[0] + K4[0]) + (1/3)*(K2[0]+K3[0]), \
           V + (1/6)*(K1[1] + K4[1]) + (1/3)*(K2[1]+K3[1]), \
           I + (1/6)*(K1[2] + K4[2]) + (1/3)*(K2[2]+K3[2]), \
           H + (1/6)*(K1[3] + K4[3]) + (1/3)*(K2[3]+K3[3]), \
           R + (1/6)*(K1[4] + K4[4]) + (1/3)*(K2[4]+K3[4]), \
           D + (1/6)*(K1[5] + K4[5]) + (1/3)*(K2[5]+K3[5])
    
      
# ------------ JOINT LOSS FUNCTIONS --------------------------
def loss_function(S_data, V_data, I_data, H_data, R_data, D_data, \
                  beta_model_all, eta_model_all, gammai_model_all, gammah_model_all, deltai_model_all, deltah_model_all, vrate_model_all):
    RK4_result = RK4(S_data[0:-1], V_data[0:-1], I_data[0:-1], H_data[0:-1], R_data[0:-1], D_data[0:-1], \
                  beta_model_all, eta_model_all, gammai_model_all, gammah_model_all, deltai_model_all, deltah_model_all, vrate_model_all, Dt)
    
    loss_S = (10**(0))*( torch.mean((S_data[1:] - RK4_result[0])**2))
    
    loss_V = (10**(0))*( torch.mean((V_data[1:] - RK4_result[1])**2))
    
    loss_I = (10**(2))*( torch.mean((I_data[1:] - RK4_result[2])**2))
    
    loss_H = 2*(10**(3))*( torch.mean((H_data[1:] - RK4_result[3])**2))

    loss_R = 3*(10**(2))*( torch.mean((R_data[1:] - RK4_result[4])**2))
    
    loss_D = (10**(6))*( torch.mean((D_data[1:] - RK4_result[5])**2))
    
    total_loss = (loss_S + loss_V + loss_I + loss_H +  loss_R + loss_D)

    return total_loss, loss_S, loss_V, loss_I, loss_H, loss_R, loss_D
# --------------- ENDS HERE ---------------------------------------------


## ------------------ FUNCTION FOR TRAINING THE NEURAL NETWORKS ---------------------------------------------------
error_toleration = 7*(10**(-10))
def train_SVIHRD_model(max_epoch, beta_model, eta_model, gammai_model, gammah_model, deltai_model, deltah_model, vrate_model):

    optimizer = optim.Adam(list(beta_model.parameters()) + list(gammai_model.parameters()) + list(gammah_model.parameters()) \
                           + list(deltai_model.parameters()) + list(deltah_model.parameters()) \
                           + list(eta_model.parameters())+ list(vrate_model.parameters()), lr=0.04)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

    total_loss_values = []
    log_loss_values = []
    log_loss_values_S = []
    log_loss_values_V = []
    log_loss_values_I = []
    log_loss_values_H = []
    log_loss_values_R = []
    log_loss_values_D = []
    rho_values = []

    for epoch in range(max_epoch):
        optimizer.zero_grad();

        features = input_tensor
        beta_model_all = beta_model(features);
        eta_model_all = eta_model(features);
        gammai_model_all = gammai_model(features);
        gammah_model_all = gammah_model(features);
        deltai_model_all = deltai_model(features);
        deltah_model_all = deltah_model(features);
        vrate_model_all = vrate_model(features);


        if epoch == 0 or (not loss_balancing):
            
            loss, loss_S, loss_V, loss_I, loss_H, loss_R, loss_D = loss_function(S_data, V_data, I_data, H_data, R_data, D_data, \
                                                                                 beta_model_all, eta_model_all, gammai_model_all, gammah_model_all, \
                                                                                 deltai_model_all, deltah_model_all, vrate_model_all)
            loss_S_init = loss_S.item(); loss_V_init = loss_V.item(); loss_I_init = loss_I.item(); loss_H_init = loss_H.item(); loss_R_init = loss_R.item(); loss_D_init = loss_D.item();
            loss_S_prev = loss_S.item(); loss_V_prev = loss_V.item(); loss_I_prev = loss_I.item(); loss_H_prev = loss_H.item();  loss_R_prev = loss_R.item();  loss_D_prev = loss_D.item();
           
            lambda_S = 1; lambda_V = 1; lambda_I = 1; lambda_H = 1; lambda_R = 1; lambda_D = 1;

            true_loss = loss.item()
            loss.backward()

        else:

            loss, loss_S, loss_V, loss_I, loss_H, loss_R, loss_D = loss_function(S_data, V_data, I_data, H_data, R_data, D_data, \
                                                                                 beta_model_all, eta_model_all, gammai_model_all, gammah_model_all, \
                                                                                 deltai_model_all, deltah_model_all, vrate_model_all)
            true_loss = loss.item()


            ## ------------------  ReLoBRaLo IMPLEMENTATION ------------------------------------------
            alpha = 0.75
            Temp = 20000
            exp_rho = 999/1000
            rho = bernoulli(p=exp_rho)
            rho_values.append(rho)
            
            #computing lambda_balance_t_ZERO
            exp_bal_S = (torch.exp(loss_S/(Temp*loss_S_init))).item(); exp_bal_V = (torch.exp(loss_V/(Temp*loss_V_init))).item(); exp_bal_I = (torch.exp(loss_I/(Temp*loss_I_init))).item()
            exp_bal_H = (torch.exp(loss_H/(Temp*loss_H_init))).item(); exp_bal_R = (torch.exp(loss_R/(Temp*loss_R_init))).item(); exp_bal_D = (torch.exp(loss_D/(Temp*loss_D_init))).item()

            sum_exp_bal = (exp_bal_S+exp_bal_V+exp_bal_I+exp_bal_H+exp_bal_R+exp_bal_D)

            lambda_bal_S_LB = exp_bal_S/sum_exp_bal; lambda_bal_V_LB = exp_bal_V/sum_exp_bal; lambda_bal_I_LB = exp_bal_I/sum_exp_bal
            lambda_bal_H_LB = exp_bal_H/sum_exp_bal; lambda_bal_R_LB = exp_bal_R/sum_exp_bal; lambda_bal_D_LB = exp_bal_D/sum_exp_bal#
            
            #computing lambda_balance_t_t-1
            exp_bal_S = (torch.exp(loss_S/(Temp*loss_S_prev))).item(); exp_bal_V = (torch.exp(loss_V/(Temp*loss_V_prev))).item(); exp_bal_I = (torch.exp(loss_I/(Temp*loss_I_prev))).item()
            exp_bal_H = (torch.exp(loss_H/(Temp*loss_H_prev))).item(); exp_bal_R = (torch.exp(loss_R/(Temp*loss_R_prev))).item(); exp_bal_D = (torch.exp(loss_D/(Temp*loss_D_prev))).item()


            sum_exp_bal = (exp_bal_S+exp_bal_V+exp_bal_I+exp_bal_H+exp_bal_R+exp_bal_D)

            
            lambda_bal_S = exp_bal_S/sum_exp_bal; lambda_bal_V = exp_bal_V/sum_exp_bal; lambda_bal_I = exp_bal_I/sum_exp_bal
            lambda_bal_H = exp_bal_H/sum_exp_bal; lambda_bal_R = exp_bal_R/sum_exp_bal; lambda_bal_D = exp_bal_D/sum_exp_bal


            lambda_hist_S = rho*lambda_S + (1- rho)*lambda_bal_S_LB
            lambda_hist_V = rho*lambda_V + (1- rho)*lambda_bal_V_LB
            lambda_hist_I = rho*lambda_I + (1- rho)*lambda_bal_I_LB
            lambda_hist_H = rho*lambda_H + (1- rho)*lambda_bal_H_LB
            lambda_hist_R = rho*lambda_R + (1- rho)*lambda_bal_R_LB
            lambda_hist_D = rho*lambda_D + (1- rho)*lambda_bal_D_LB

            lambda_S = alpha*lambda_hist_S + (1-alpha)*lambda_bal_S
            lambda_V = alpha*lambda_hist_V + (1-alpha)*lambda_bal_V
            lambda_I = alpha*lambda_hist_I + (1-alpha)*lambda_bal_I
            lambda_H = alpha*lambda_hist_H + (1-alpha)*lambda_bal_H
            lambda_R = alpha*lambda_hist_R + (1-alpha)*lambda_bal_R
            lambda_D = alpha*lambda_hist_D + (1-alpha)*lambda_bal_D


            loss = lambda_S*loss_S + lambda_V*loss_V + lambda_I*loss_I + lambda_H*loss_H + lambda_R*loss_R + lambda_D*loss_D 
            ## ------------------  ENDS HERE ------------------------------------------


            loss.backward()

            loss_S_prev = loss_S.item();  loss_V_prev = loss_V.item(); loss_I_prev = loss_I.item();  loss_H_prev = loss_H.item();  loss_R_prev = loss_R.item();  loss_D_prev = loss_D.item();
            

        optimizer.step()
        scheduler.step()

        loss, loss_S, loss_V, loss_I, loss_H, loss_R, loss_D = loss_function(S_data, V_data, I_data, H_data, R_data, D_data, \
                                                                             beta_model_all, eta_model_all, gammai_model_all, gammah_model_all, \
                                                                             deltai_model_all, deltah_model_all, vrate_model_all)

        total_loss_values.append(loss.item())
        log_loss_values.append(np.log(loss.item()))
        log_loss_values_S.append(torch.log(loss_S).item())
        log_loss_values_V.append(torch.log(loss_V).item())
        log_loss_values_H.append(torch.log(loss_H).item())
        log_loss_values_I.append(torch.log(loss_I).item())
        log_loss_values_R.append(torch.log(loss_R).item())
        log_loss_values_D.append(torch.log(loss_D).item())

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Total Loss:{loss.item()}")


        if (loss.item() < error_toleration) :
            break

    print(f"lossS:{loss_S.item()},lossV:{loss_V.item()},lossI:{loss_I.item()},lossH:{loss_H.item()},lossR:{loss_R.item()},lossD:{loss_D.item()}, total:{loss.item()}")
    
    return total_loss_values, log_loss_values, log_loss_values_S, log_loss_values_V, log_loss_values_I, log_loss_values_H, log_loss_values_R, log_loss_values_D, rho_values
#####--------------------------------------- FUNCTION END HERE ---------------------------------------


##---------------- TRAIN THE NEURAL NETWORK -------------------------
n_epoch = 50000
loss_list, log_loss_list, log_loss_list_S, log_loss_list_V, log_loss_list_I, log_loss_list_H, log_loss_list_R, log_loss_list_D, rho_list = train_SVIHRD_model(n_epoch, beta_model, eta_model, gammai_model, gammah_model, deltai_model, deltah_model, vrate_model)
#----------------- TRAIN ENDS HERE -----------------

beta_model.eval(); eta_model.eval(); gammai_model.eval(); gammah_model.eval(); deltai_model.eval(); deltah_model.eval(); vrate_model.eval()
beta = beta_model(input_tensor).detach().numpy();
eta = eta_model(input_tensor).detach().numpy();
gammai = gammai_model(input_tensor).detach().numpy();
gammah = gammah_model(input_tensor).detach().numpy();
deltai = deltai_model(input_tensor).detach().numpy();
deltah = deltah_model(input_tensor).detach().numpy();
vrate = vrate_model(input_tensor).detach().numpy();


##Solve the SVIHRD numerically using the estimated rates obtained by training NN
S = [S_data[0].detach().numpy()]; V = [V_data[0].detach().numpy()]; I = [I_data[0].detach().numpy()]; H = [H_data[0].detach().numpy()]; R = [R_data[0].detach().numpy()]; D = [D_data[0].detach().numpy()];
for i in range(n-1):
    K1 = F(S[i],V[i],I[i],H[i],R[i],D[i], beta[i],eta[i],gammai[i],gammah[i], deltai[i],deltah[i], vrate[i],dt)
    K2 = F(S[i]+K1[0]/2, V[i]+K1[1]/2, I[i]+K1[2]/2, H[i]+K1[3]/2, R[i]+K1[4]/2, D[i]+K1[5]/2, beta[i],eta[i],gammai[i],gammah[i], deltai[i],deltah[i], vrate[i],dt)
    K3 = F(S[i]+K2[0]/2, V[i]+K2[1]/2, I[i]+K2[2]/2, H[i]+K2[3]/2, R[i]+K2[4]/2, D[i]+K2[5]/2, beta[i],eta[i],gammai[i],gammah[i], deltai[i],deltah[i], vrate[i],dt)
    K4 = F(S[i]+K3[0], V[i]+K3[1], I[i]+K3[2], H[i]+K3[3], R[i]+K3[4], D[i]+K3[5], beta[i],eta[i],gammai[i],gammah[i], deltai[i],deltah[i], vrate[i],dt)
    
    S.append( S[i] + (1/6)*(K1[0] + K4[0]) + (1/3)*(K2[0]+K3[0]) );
    V.append( V[i] + (1/6)*(K1[1] + K4[1]) + (1/3)*(K2[1]+K3[1]) );
    I.append( I[i] + (1/6)*(K1[2] + K4[2]) + (1/3)*(K2[2]+K3[2]) );
    H.append( H[i] + (1/6)*(K1[3] + K4[3]) + (1/3)*(K2[3]+K3[3]) );
    R.append( R[i] + (1/6)*(K1[4] + K4[4]) + (1/3)*(K2[4]+K3[4]) );
    D.append( D[i] + (1/6)*(K1[5] + K4[5]) + (1/3)*(K2[5]+K3[5]) );

##Plot the Model Fitting
fig,ax = plt.subplots(2,3);

ax[0][0].plot(t_data, N_p*S_data.detach().numpy(), '-', color='orange', lw=3);ax[0][0].plot(t_data, np.array(S)*N_p,'-',color='blue',lw=1);
ax[0][0].set_xlim(t_data[0], t_data[-1]); ax[0][0].set_xlabel(r"$t$"); ax[0][0].set_title(r"$S(t)$ v.s. $S_{model}(t)$", fontsize=12);

ax[0][1].plot(t_data, N_p*I_data.detach().numpy(), '-', color='orange', lw=3); ax[0][1].plot(t_data, np.array(I)*N_p,'-',color='blue',lw=1);
ax[0][1].set_xlim(t_data[0], t_data[-1]); ax[0][1].set_xlabel(r"$t$"); ax[0][1].set_title(r"$I(t)$ v.s. $I_{model}(t)$", fontsize=12);

ax[0][2].plot(t_data, N_p*D_data.detach().numpy(), '-', color='orange', lw=3); ax[0][2].plot(t_data, np.array(D)*N_p,'-',color='blue',lw=1);
ax[0][2].set_xlim(t_data[0], t_data[-1]); ax[0][2].set_xlabel(r"$t$"); ax[0][2].set_title(r"$D(t)$ v.s. $D_{model}(t)$", fontsize=12);

ax[1][0].plot(t_data, N_p*H_data.detach().numpy(), '-', color='orange', lw=3); ax[1][0].plot(t_data, np.array(H)*N_p,'-',color='blue',lw=1);
ax[1][0].set_xlim(t_data[0], t_data[-1]); ax[1][0].set_xlabel(r"$t$"); ax[1][0].set_title(r"$H(t)$ v.s. $H_{model}(t)$", fontsize=12);

ax[1][1].plot(t_data, N_p*R_data.detach().numpy(), '-', color='orange', lw=3); ax[1][1].plot(t_data, np.array(R)*N_p,'-',color='blue',lw=1);
ax[1][1].set_xlim(t_data[0], t_data[-1]); ax[1][1].set_xlabel(r"$t$"); ax[1][1].set_title(r"$R(t)$ v.s. $R_{model}(t)$", fontsize=12);

ax[1][2].plot(t_data, N_p*V_data.detach().numpy(), '-', color='orange', lw=3); ax[1][2].plot(t_data, np.array(V)*N_p,'-',color='blue',lw=1);
ax[1][2].set_xlim(t_data[0], t_data[-1]); ax[1][2].set_xlabel(r"$t$"); ax[1][2].set_title(r"$V(t)$ v.s. $V_{model}(t)$", fontsize=12);

fig.tight_layout(pad=0.5); fig.show();


### Plot the Estimated Rates
fig,ax=plt.subplots(1,3,figsize=(9,3));
ax[0].plot(t_data[0:-1], beta/N_p, color='blue',lw=1); ax[0].set_title(r"$\beta_{\Delta t}/N_{P}$", fontweight='heavy');
ax[1].plot(t_data[0:-1], eta/N_p, color='blue',lw=1); ax[1].set_title(r"$\eta_{\Delta t}/N_{p}$", fontweight='heavy');
ax[2].plot(t_data[0:-1], vrate, color='blue',lw=1); ax[2].set_title(r"$v_{\Delta t}$", fontweight='heavy');
fig.tight_layout(pad=0.5); fig.show(); 

fig,ax=plt.subplots(2,2,figsize=(8,4));
ax[0][0].plot(t_data[0:-1], gammai, color='blue', lw=1); ax[0][0].set_title(r"$\gamma_{i,\Delta t}$", fontweight='heavy');
ax[0][1].plot(t_data[0:-1], gammah, color='blue', lw=1); ax[0][1].set_title(r"$\gamma_{h,\Delta t}$", fontweight='heavy');
ax[1][0].plot(t_data[0:-1], deltai, color='blue', lw=1); ax[1][0].set_title(r"$\delta_{i,\Delta t}$", fontweight='heavy');
ax[1][1].plot(t_data[0:-1], deltah, color='blue', lw=1); ax[1][1].set_title(r"$\delta_{h,\Delta t}$", fontweight='heavy');
fig.tight_layout(pad=0.5); fig.show(); 


## Plot the Loss Functions
fig,ax=plt.subplots(2,3);
ax[0][0].plot(log_loss_list_S); ax[0][0].plot(rho_list); ax[0][0].set_title("loss S");
ax[0][1].plot(log_loss_list_V); ax[0][1].plot(rho_list); ax[0][1].set_title("loss V");
ax[0][2].plot(log_loss_list_I); ax[0][2].plot(rho_list); ax[0][2].set_title("loss I");
ax[1][0].plot(log_loss_list_H); ax[1][0].plot(rho_list); ax[1][0].set_title("loss H");
ax[1][1].plot(log_loss_list_R); ax[1][1].plot(rho_list); ax[1][1].set_title("loss R");
ax[1][2].plot(log_loss_list_D); ax[1][2].plot(rho_list); ax[1][2].set_title("loss D");
fig.show()



%% check behaviour of LipSDP for different values of alpha and b = 1:
clear all;
close all;

% Params
alpha = 0;
beta = 1;
n_layers = 6;
n_neurons_perlayer = 4;
lw = 1.7

net_dim = [n_neurons_perlayer];
for i = 1:n_layers
    net_dim(end+1) = n_neurons_perlayer;
end
net_dim(end+1) = n_neurons_perlayer;

%weights chosen uniformly between -1 and 1, also calculate trivial upper
% and Lipschitz of linear Network
weights = {};
% matrix for linear network
W_ges = eye(n_neurons_perlayer);
tr_upper = 1;
time_trivialUpper = 0;
for k = 1:(n_layers+1)
    W = 2*rand(n_neurons_perlayer,n_neurons_perlayer)-ones(n_neurons_perlayer, n_neurons_perlayer);
    weights{k} = W;
    W_ges = W*W_ges;
    tic;
    tr_upper = tr_upper*operatorNorm(W);
    time_mult = toc;
    time_trivialUpper = time_trivialUpper + time_mult;
end
L_linear = operatorNorm(W_ges)

tic;
LipSDP_neu = LipSDP_withAlphaBeta_callFromMatlab(weights, net_dim, 'neuron', alpha, beta);
time_SDPneuron = toc;

tic;
LipSDP_lay = LipSDP_withAlphaBeta_callFromMatlab(weights, net_dim, 'layer', alpha, beta);
time_SDPlayer = toc;  

alpha_combettes = 1/2 -1/2*alpha;
tic;
L_combettes = computeCombettes(weights,net_dim, alpha_combettes);
time_Cpsup = toc;

tic;
L_combettes_large = computeCombettes2(weights,net_dim, alpha_combettes);
time_Cpsum = toc;


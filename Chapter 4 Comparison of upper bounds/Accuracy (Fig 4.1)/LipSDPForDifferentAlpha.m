%% check behaviour of LipSDP for different values of alpha and b = 1:
clear all;
close all;

% Params
beta = 1;
n_layers = 4;
n_neurons_perlayer = 4;
lw = 1.7

alpha_array = linspace(-1,1,100);
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
for k = 1:(n_layers+1)
    W = 2*rand(n_neurons_perlayer,n_neurons_perlayer)-ones(n_neurons_perlayer, n_neurons_perlayer);
    weights{k} = W;
    W_ges = W*W_ges;
    tr_upper = tr_upper*operatorNorm(W);
end
L_linear = operatorNorm(W_ges)

% compute LipSDP for different values of alpha
LipSDP_neuron_array = [];
LipSDP_layer_array = [];
Combettes_array = [];
Combettes_array2 = [];

for alpha = alpha_array
    LipSDP_neu = LipSDP_withAlphaBeta_callFromMatlab(weights, net_dim, 'neuron', alpha, beta);
    LipSDP_neuron_array(end+1) = LipSDP_neu.L;

    LipSDP_lay = LipSDP_withAlphaBeta_callFromMatlab(weights, net_dim, 'layer', alpha, beta);
    LipSDP_layer_array(end+1) = LipSDP_lay.L;    
    
    alpha_combettes = 1/2 -1/2*alpha;
    L_combettes = computeCombettes(weights,net_dim, alpha_combettes);
    Combettes_array(end+1) = L_combettes;

    L_combettes_large = computeCombettes2(weights,net_dim, alpha_combettes);
    Combettes_array2(end+1) = L_combettes_large;
end

% create Plots
figure
hold on;
grid on;

plot([-1,1],[tr_upper,tr_upper],'LineWidth',lw, 'Color','#77AC30');
plot(alpha_array, LipSDP_neuron_array,'LineWidth',lw, 'Color', '#0072BD');
plot(alpha_array, LipSDP_layer_array,'LineWidth',lw, 'Color', '#7E2F8E');
plot(alpha_array, Combettes_array,'LineWidth',lw,'Color', '#D95319');
plot(alpha_array, Combettes_array2,'LineWidth',lw,'Color', '#EDB120');
plot([-1,1],[L_linear,L_linear],'k','LineWidth',lw);
ylim([L_linear-1,tr_upper+1]);
xlabel("alpha");
ylabel("upper bound Lipschitz constant");
lgd = legend("$L_{naive}$", "$L_{SDP}^{neu}$", "$L_{SDP}^{lay}$", "$L_{CP}^{sup}$","$L_{CP}^{sum}$", "LC of linear network {\cal L}_2^{lin}", 'interpreter','latex');
lgd.Location = 'northeast';
lgd.FontSize = 12;
pos = get(lgd,'Position');
disp(pos);
posx = 0.5360 ;
posy = 0.56;
set(lgd,'Position',[posx posy pos(3) pos(4)]);
xlabel('$\alpha$', 'interpreter','latex')
ylabel('upper bound','interpreter','latex')
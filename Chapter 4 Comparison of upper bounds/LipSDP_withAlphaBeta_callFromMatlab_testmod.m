% Upper bound by LipSDP-Neuron and layer
% parameters: weight-mastrices, x: array of net dimensions, s: string containing 'neuron' or 'layer'.

function Lip = LipSDP(W, x, s, alpha, beta)

addpath(genpath('D:\Uni\UniStuttgart\SS 22\Bachelorarbeit\YALMIP-master'))
addpath(genpath('C:\Program Files\Mosek\9.3'))


net_dim = x;
l = length(net_dim)-1;
eps = 10^(-9);

if strcmp(s,'neuron')
    T = diag(sdpvar(sum(net_dim(2:l)),1));
elseif strcmp(s,'layer')
    for layer = 2:l
        if layer == 2
            T = diag(sdpvar(1)*ones(net_dim(layer),1));
            disp(T)
        else

            T = blkdiag(T,diag(sdpvar(1)*ones(net_dim(layer),1)));
            disp(T)
        end
    end
end
rho = sdpvar(1);

A = [blkdiag(W{1:l-1}), zeros(sum(net_dim(2:l)), net_dim(l))];
B = [zeros(sum(net_dim(2:l)), net_dim(1)), eye(sum(net_dim(2:l)))];
% C = [zeros(net_dim(length(net_dim)), sum(net_dim(1:l-1))), W{l}];

Q = blkdiag(-rho*eye(net_dim(1)),zeros(sum(net_dim(2:l-1))),W{l}'*W{l});
M = [A; B]' * [-2*alpha*beta*T, (alpha+beta)*T; (alpha+beta)*T, -2*T] * [A; B] + Q;
F1 = (M <= -eps*eye(sum(net_dim(1:l))));

optimize(F1, rho);
Lip.L = sqrt(double(rho));
Lip.T = T;
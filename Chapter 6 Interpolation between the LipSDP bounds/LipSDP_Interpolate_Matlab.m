% Upper bound by LipSDP Interpolation

function [Lip, time_solver, time_overhead_total] = LipSDP_Interpolate_Matlab(net_dim)

% Path to YALMIP and Mosek
%addpath(genpath('D:\Uni\UniStuttgart\SS 22\Bachelorarbeit\YALMIP-master'))
addpath(genpath('D:\Lukis zeug\YALMIP\YALMIP-master'))
addpath(genpath('C:\Program Files\Mosek\9.3'))

l = length(net_dim)-1;
eps = 10^(-9);

fname = 'weights.json';
weights = jsondecode(fileread(fname));
fn = fieldnames(weights);
for k=1:numel(fn)
    W{k}= weights.(fn{k});
end

fname = 'X_dict.json';
x_read = jsondecode(fileread(fname));
fn = fieldnames(x_read);
for k=1:numel(fn)
    X{k}= x_read.(fn{k});
end

tic;
% construct T according to solution of MILP X
for layer = 1:l-1
    T_layer = zeros(net_dim(layer));
    x_lay = X{layer};    
    for dof = 1:length(x_lay(:,1))
        T_layer = T_layer + diag(sdpvar(1)*x_lay(dof,:));
    end
    if layer == 1
        T = T_layer;
    else
        T = blkdiag(T,T_layer);
    end
end
rho = sdpvar(1);

A = [blkdiag(W{1:l-1}), zeros(sum(net_dim(2:l)), net_dim(l))];
B = [zeros(sum(net_dim(2:l)), net_dim(1)), eye(sum(net_dim(2:l)))];

Q = blkdiag(-rho*eye(net_dim(1)),zeros(sum(net_dim(2:l-1))),W{l}'*W{l});
M = [A; B]' * [zeros(sum(net_dim(2:l))), T; T, -2*T] * [A; B] + Q;
%F1 = (M <= -eps*eye(sum(net_dim(1:l))));
F1 = (M <= 0);


% options = sdpsettings('solver','MOSEK','mosek.MSK_IPAR_NUM_THREADS', 1);
options = sdpsettings('solver','MOSEK');
sol = optimize(F1, rho, options);
time_overhead_total = toc;
time_solver = sol.solvertime;
Lip = sqrt(double(rho));
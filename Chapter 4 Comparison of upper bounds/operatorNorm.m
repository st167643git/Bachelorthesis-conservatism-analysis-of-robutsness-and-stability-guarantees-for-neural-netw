%% compute operatornorm
function Norm = operatorNorm(A_arg)
     Norm = sqrt((max(eig(A_arg'*A_arg))));
end
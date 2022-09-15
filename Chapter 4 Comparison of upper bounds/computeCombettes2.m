%% Implementation of the higher upper bound by combettes.
function upper_bound = computeCombettes2(weights,net_dim, alpha)
    
    hidden_layer_dim = net_dim(2:end-1);
    n_hidden_neurons = sum(net_dim(2:end-1));
    n_hidden_layers = length(net_dim) -2;

    J = 1:n_hidden_layers;
    W_part = weights{1};
    for p = 2:length(weights)
        W_part = weights{p}*W_part;
    end
    upper_bound = (1-alpha)^n_hidden_layers*operatorNorm(W_part);
    for k = 1:n_hidden_layers
        boundary_array = nchoosek(J,k);
        for j = 1:length(boundary_array(:,1))
            boundaries = boundary_array(j,:);
            boundaries(end+1) = n_hidden_layers+1;
            next_val = 1;
            W_part = 0;
            product = 1;
            for boundary = boundaries
                while next_val <= boundary
                    if W_part == 0
                        W_part = weights{next_val};
                    else
                        W_part = weights{next_val}*W_part;
                    end
                    next_val = next_val +1;
                end
                %disp(W_part)
                product = product*operatorNorm(W_part);
                W_part = 0;
            
            end
            upper_bound = upper_bound + (1-alpha)^(n_hidden_layers-k)*(alpha^k)*product;
        end
    end

    
 



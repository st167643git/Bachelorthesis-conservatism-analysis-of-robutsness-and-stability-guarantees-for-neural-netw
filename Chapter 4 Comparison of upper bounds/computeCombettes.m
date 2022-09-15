function theta_m = Combettes(weights,net_dim, alpha)
    
    hidden_layer_dim = net_dim(2:end-1);
    n_hidden_neurons = sum(net_dim(2:end-1));
    n_hidden_layers = length(net_dim) -2;
    val = 1-2*alpha;

    N = n_hidden_neurons;
    comb_matrix = dec2bin(0:2^N-1)' - '0';
    
    theta_m = 0;
    for k = 1:2^n_hidden_neurons
        pattern = comb_matrix(:,k);  

        last_value = 1;
        W_ges = weights{1};
        
        for n_layer = 1: n_hidden_layers
            pattern_part1 = pattern(last_value:last_value+hidden_layer_dim(n_layer)-1);
        
            pattern_part2 = ones(hidden_layer_dim(n_layer),1) - pattern_part1;
            
            pattern_part = val*pattern_part1 + pattern_part2;
            
            Lambda = diag(pattern_part);
            last_value = last_value + hidden_layer_dim(n_layer);
            W_ges = Lambda*W_ges;
            W_next = weights{n_layer +1};
            W_ges = W_next*W_ges;
        end

        Norm_linear_subregion = operatorNorm(W_ges);

        if Norm_linear_subregion > theta_m
            theta_m = Norm_linear_subregion;
        end
    end
end


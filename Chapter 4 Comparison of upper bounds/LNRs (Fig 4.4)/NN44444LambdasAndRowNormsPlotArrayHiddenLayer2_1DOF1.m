%% Erstellung Plot_array mit gemittelten Lambda-Werten bei verschiedenen SpaltenNormen in der zweiten hidden layer für 44444 NN.

clear all;
close all;

addpath('C:\Program files\Mosek\9.3\toolbox\r2015a');

n0 = 4;
n1 = 4;
n2 = 4;
n3 = 4;
n4 = 4;


Plot_array = {};
counterm = 1;

for m = linspace(-2,2,21)
    countern = 1;

    for n = linspace(-2,2,21)
        
        lambda_array = [];
        rel_error_array = [];
    
        for i = 1:1
            v1 = VectorFixedNorm(n3,10^(0.5*m));
            v2 = VectorFixedNorm(n3,10^(0.5*n));
            v3 = VectorFixedNorm(n3,10^(-0.5*m))';
            v4 = VectorFixedNorm(n3,10^(-0.5*n))';
            
            W0 = rand(4,4)-0.5;
            W1 = rand(4,4)-0.5;
            W2 = rand(4,4)-0.5;
            W3 = rand(4,4)-0.5;        
  
            W2(:,1) = v1;
            W2(:,2) = v2;
            W1(1,:) = v3;
            W1(2,:) = v4;
    
            A = blkdiag(W0,W1,W2,W3);
            A = A(1:(n1+n2+n3),:);
            B = blkdiag(eye(n0),eye(n1), eye(n2),eye(n3));
            B = B(n0+1:end,:);
            C = [zeros(n4,n0+n1+n2) W3];      
        
            Ls1 = sdpvar(1);
            Ls2 = sdpvar(1);
            la1 = sdpvar(1);
            la2 = sdpvar(1);
            la3 = sdpvar(1);
            la4 = sdpvar(1);
            la11 = sdpvar(1);
            la12 = sdpvar(1);
            la13 = sdpvar(1);
            la14 = sdpvar(1);
            la21 = sdpvar(1);
            la22 = sdpvar(1);
            la23 = sdpvar(1);
            la24 = sdpvar(1);
            la31 = sdpvar(1);
            la32 = sdpvar(1);
            la33 = sdpvar(1);
            la34 = sdpvar(1);
                   
            T = diag([la11,la12,la13,la14,la21,la22,la23,la24,la31,la32,la33,la34]');
            M1 = [A;B]'*[0*T T; T -2*T]*[A;B];
            M2 = blkdiag(-Ls1*eye(n0), zeros(n1,n1), zeros(n2,n2), W3'*W3);
            Mn = -(M1+M2)
            F = [Mn >= 0, la11 >= 0, la12 >= 0, la13 >= 0, la14 >= 0, la21 >= 0, la22 >= 0, la23 >= 0, la24 >= 0, la31 >= 0, la32 >= 0, la33 >= 0, la34 >= 0];
         
            options = sdpsettings('solver', 'MOSEK');
            obj = Ls1;
            soln = optimize(F,obj,options);
            Ln1 = value(Ls1);
    
            lan21 = value(la21);
            lan22 = value(la22);
            lan23 = value(la23);
            lan24 = value(la24); 
    
            T = diag([la11,la12,la13,la14,la21,la21,la22,la23,la31,la32,la33,la34]');
            M1 = [A;B]'*[0*T T; T -2*T]*[A;B];
            M2 = blkdiag(-Ls2*eye(n0), zeros(n1,n1), zeros(n2,n2), W3'*W3);
            Mn = -(M1+M2)
            F = [Mn >= 0, la11 >= 0, la12 >= 0, la13 >= 0, la14 >= 0, la21 >= 0, la22 >= 0, la23 >= 0, la31 >= 0, la31 >= 0, la33 >= 0, la34 >= 0];    
    
            obj = Ls2;
            soln2 = optimize(F,obj,options);
            Ln2 = value(Ls2);
    
            lan21con = value(la21);
            lan22con = value(la22);
            lan23con = value(la23);
    
    
            lambda_array(1,end+1) = lan21;
            lambda_array(2,end) = lan22;
            lambda_array(3,end) = lan23;
            lambda_array(4,end) = lan24;
            
            % lambda von LipSDP-Layer gespeichert an fünfter Stelle
            lambda_array(5,end+1) = lan21con;
            lambda_array(6,end) = lan22con;
            lambda_array(7,end) = lan23con;
            rel_error_array(1,end+1) = (sqrt(Ln2)-sqrt(Ln1))/sqrt(Ln1);
        end
    
        lan21_mean = mean(lambda_array(1,:));
        lan22_mean = mean(lambda_array(2,:));
        lan23_mean = mean(lambda_array(3,:));
        lan24_mean = mean(lambda_array(4,:));
        
        lan21con_mean = mean(lambda_array(5,:));
        lan22con_mean = mean(lambda_array(6,:));
        lan23con_mean = mean(lambda_array(7,:));
      
        rel_error_mean = mean(rel_error_array);
    
        Plot_array{countern,counterm}.n = n;
        Plot_array{countern,counterm}.m = m;
        Plot_array{countern,counterm}.lan1neu_mean = lan21_mean;
        Plot_array{countern,counterm}.lan2neu_mean = lan22_mean;
        Plot_array{countern,counterm}.lan3neu_mean = lan23_mean;
        Plot_array{countern,counterm}.lan4neu_mean = lan24_mean;
        Plot_array{countern,counterm}.lan1con_mean = lan21con_mean;
        Plot_array{countern,counterm}.lan2con_mean = lan22con_mean;
        Plot_array{countern,counterm}.lan3con_mean = lan23con_mean;
        Plot_array{countern,counterm}.rel_error_mean = rel_error_mean;
        
        countern = countern +1;
    end
    counterm = counterm +1;
end

save("Plot_array_44444NN_HiddenLayer2_1DOF_writing2.mat","Plot_array");
%% Plot Erstellung zu relativen Fehlern zu Plot_Array
clear all;
close all,

load Plot_array_44444NN_HiddenLayer2_1DOF_writing2.mat;

n_val = [];
m_val = [];

hold on;
%set(gca, 'YScale', 'log')

err_array = [];

for i = 1:length(Plot_array)
    for j = 1:length(Plot_array)
        n_val(end+1) = Plot_array{i,j}.n;
        m_val(end+1) = Plot_array{i,j}.m;
        err_array(end+1) = Plot_array{i,j}.rel_error_mean;
    end
end

nv = linspace(-2,2,21);
mv = linspace(-2,2,21);
[N,M] = meshgrid(nv, mv);
Z = griddata(n_val,m_val,err_array,N,M);

figure(1);
surf(N, M, Z);
xlabel('x');
ylabel('y');
%zlabel("relative error");
%title('relative error 44444NN with a single constrained DOF for different values of a and b');
lgd = legend("$\frac{L_{SDP}^{neu}-L_{SDP}^{res}}{L_{SDP}^{res}}$","Interpreter","latex");
lgd.Location = 'northeast';
lgd.FontSize = 14;
pos = get(lgd,'Position');
posx = 0.5360 ;
posy = 0.56;
set(lgd,'Position',[posx posy pos(3) pos(4)]);
grid on;

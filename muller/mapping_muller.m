clc; close all; clear

A = [-200,-100,-175,15];
a = [-1,-1,-6.5,0.7];
b = [0,0,11,0.6];
c = [-10,-10,-6.5,0.7];
x0 = [1,0,-0.5,-1];
y0 = [0,0.5,1.5,1];

X = -2:0.1:1.2;
Y = -0.8:0.1:2.5;
[x,y] = meshgrid(X,Y);

%% muller-Brown potential function
f = @computeSubTerm;
sums = 0;
for i = 1:4
    sums = sums + f(x, y, x0(i), y0(i), A(i), a(i), b(i), c(i));
end
Z = sums- min(min(sums));
Z(Z>10000) = inf;

%% plotting
colormap('default');
contourf(X,Y,Z,100)
    
%% subfunction
function sums = computeSubTerm(x,y, x0,y0, A, a, b, c)
    sums = A.*exp( a.*(x-x0).^2 + b.*(x-x0).*(y-y0) + c.*(y-y0).^2 );
end
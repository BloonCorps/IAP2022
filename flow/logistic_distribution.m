clc; close all; clear

x = -10:0.01:10;
mu = mean(x);

%% muller-Brown potential function
g = @gaussian;
l = @logistic;
sigma = 1.0;

y1 = g(x, mu, sigma);
y2 = l(x, mu, sigma);

%% plotting
figure(1); hold on
plot(x, y1)
plot(x, y2)
    
%% subfunction
function f = gaussian(x, mu, sigma)
    f = 1/(2*pi*sigma^2).*exp(-(x-mu).^2./(2*sigma^2));
end

function f = logistic(x, mu, sigma)
    f = 1/sigma*exp(-(x-mu)./sigma).*(1+exp(-(x-mu)./sigma)).^(-2);
end
% number of observations (rows)
m = 80;

% generate simulated data
t1 = rand(m,1)*2*pi;
t2 = rand(m,1)*2*pi;
r1 = 1 + randn(m,1)*.05;
r2 = .5 + randn(m,1)*.05;
x1 = 2 + r1 .* cos(t1);
y1 = 2 + r1 .* sin(t1);
x2 = 2 + r2 .* cos(t2);
y2 = 2 + r2 .* sin(t2);

x = [x1; x2];
y = [y1; y2];

figure(1)
scatter(x,y)
axis equal

figure(2)
scatter(x1,y1)
hold on
scatter(x2,y2)
hold off
axis equal

% data matrix
X = [x y];

% create kernel function
sigma = 10;
rbf = @ (x,y) exp(-norm(x - y)^2 / sigma^2);

%% Compute the kpca for X using the Gaussian kernel rbf
[V, D] = kpca(X, rbf);

function [A,D] = kpca(X, k)
% Kernel PCA
% 
% Parameters
% ----------
%   X : data matrix
%   k : kernel function
% 
% Outputs
% -------

% Center data matrix
X0 = X - mean(X,1);

% Compute the kernel (Gram) matrix.
G = gramian(X0,k);

% center the kernel matrix
nrows = size(G,1);
N = ones(nrows)/nrows;
G = G - N * G - G * N + N * G * N;

% Diagonalize the Gram matrix.
[A,D,~] = eig(G);

% Sort eigenvectors and eigenvalues
lam = diag(D);
[lam,idx] = sort(lam);
A = A(:,idx);

% Get projection in feature space.
% V phi(x) = sum alpha_i * k(x_i, x), i=1..n

end

function G = gramian(X, k)
% Compute the Gram matrix.
% Warning: This will compute the entire matrix. Not a good idea if there
% are more than 50,000 columns.

[~,ncols] = size(X);

G = zeros(ncols);

for i = 1:ncols
    for j = 1:ncols
        G(i,j) = k(X(:,i),X(:,j));
    end
end
end
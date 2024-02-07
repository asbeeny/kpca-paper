n = 125; % number of observations
m = 2;  % number of attributes (dimensions)

% generate normal samples
% X = zeros([n m]);
noise1 = randn([n 1]);
noise2 = randn([n 1]);
X = 2*noise1 + 5;
Y = .5*X + .5*noise2 + 3;
plot(X, Y, '.')
axis([0 10 0 10])
axis square

[coeff,score,latent,~,explained,mu] = pca([X Y]);

X0 = X - mu(1);
Y0 = Y - mu(2);
m = sum(X0 .* Y0) / sum(X0.^2);

%%

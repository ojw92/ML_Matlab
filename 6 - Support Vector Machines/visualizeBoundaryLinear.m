function visualizeBoundaryLinear(X, y, model)
%VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
%SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
%   learned by the SVM and overlays the data on it

w = model.w;    % w refers to theta without the theta_0
b = model.b;    % b refers to theta_0
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);   % b is multiplied by x_0 = 1, so just add b
plotData(X, y);
hold on;
plot(xp, yp, '-b'); 
hold off

end

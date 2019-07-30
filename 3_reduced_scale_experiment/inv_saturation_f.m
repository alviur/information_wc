function [x,dxdf] = inv_saturation_f(f,g,xm,epsilon);

%
% INV_SATURATION_F is the inverse of an element-wise exponential function (saturation). 
% It is good to (1) model the saturation in Wilson-Cowan recurrent networks, 
% and (2) as a crude (fixed) approximation to the luminance-brightness transform. 
% 
% This saturation is normalized and modified to have these two good properties:
% (a) Some specific input, xm (e.g. the median, the average) maps into itself: xm = f(xm).
% 
%          f(x) = sign(x)*K*|x|^g  , where the constant K=xm^(1-g)
%
%     Therefore, the inverse is:
%          x = sign(f)*((1/K)*abs(f)).^(1/g)
%
% (b) Plain exponential is modified close to the origin to avoid the singularity of
%     the derivative of saturating exponentials at zero.
%     This problem is solved by imposing a parabolic behavior below a certain
%     threshold (epsilon) and imposing the continuity of the parabola and its
%     first derivative at epsilon.
%
%         f(x) = sign(x)*K*|x|^g             for |x| > epsilon
%                sign(x)*(a*|x|+b*|x|^2)     for |x| <= epsilon
%
%      with:
%                a = (2-g)*K*epsilon^(g-1)
%                b = (g-1)*K*epsilon^(g-2)
%
%      Therefore, the inverse close to the origin ( for |f| < K*epsilon^g ) is:
%
%          x = sign(f)*(-a-sqrt(a^2+4*b*f))/(2*b);  [note that for saturations b<0 and hence positive solutions x involve that sign before the sqrt]
%
% The derivative of the inverse function is the inverse of the derivative of the direct function:
% 
%         dx/df = (g*K*|x|^(g-1))^-1   for |f| > K*epsilon^g   
%                 (a + 2*b*|x|)^-1     for |f| < K*epsilon^g  
%
% In the end, the slope at the origin depends on the constant xm (smaller for bigger xm). 
% 
% The program gives the inverse function and the derivative. For the direct see SATURATION_F.M
%
% For vector/matrix inputs x, the vector/matrix with anchor points, xm, has to be the same size as x.
%
% USE:    [x,dxdf] = inv_saturation_f(f,gamma,xm,epsilon);
%
%   f     = n*m matrix with the values 
%   gamma = exponent (scalar)
%   xm    = n*m matrix with the anchor values (in wavelet representations typically anchors will be different for different subbands)
%   epsilon = threshold (scalar). It can also be a matrix the same size as f (again different epsilons per subband, e.g. epsilon = 1e-3*x_average)
%
% EXAMPLE:
%    [x,dxdf] = inv_saturation_f(linspace(-1.1,1.1,1001),0.2,ones(1,1001),0.1);
%    figure,subplot(211),plot(linspace(-1.1,1.1,1001),x)
%    subplot(212),semilogy(linspace(-1.1,1.1,1001),dxdf)
%

s = size(f);

f = f(:);
xm=xm(:);

K = xm.^(1-g);
a = (2-g)*K.*epsilon.^(g-1);
b = (g-1)*K.*epsilon.^(g-2);

pG = find(f > K.*epsilon.^g);
pp = find((f <= K.*epsilon.^g) & (f >= 0));

nG = find(f<-K.*epsilon.^g);
np = find((f > -K.*epsilon.^g) & (f <= 0));

x = f;
dxdf=f;
dfdx=f;

if isempty(pG)==0
   %f(pG) = K(pG).*x(pG).^g;                  %   for |x| > epsilon
   x(pG) = ( (1./K(pG)).*abs(f(pG)) ).^(1/g);
   dfdx(pG) = g*K(pG).*abs(x(pG)).^(g-1);    %   for |x| > epsilon   [bigger with xm and decreases with signal]
   dxdf(pG) = 1./dfdx(pG);
end
if isempty(nG)==0
   %f(nG) = - K(pG).*abs(x(nG)).^g;           %   for |x| > epsilon
   x(nG) = -((1./K(nG)).*abs(f(nG))).^(1/g);
   dfdx(nG) = g*K(nG).*abs(x(nG)).^(g-1);    %   for |x| > epsilon   [bigger with xm and decreases with signal]
   dxdf(nG) = 1./dfdx(nG);
end

if isempty(pp)==0
   %f(pp) = (a(pp).*abs(x(pp))+b(pp).*abs(x(pp)).^2);    %        for |x| <= epsilon
   x(pp) = (-a(pp)+sqrt(a(pp).^2+4*b(pp).*f(pp)))./(2*b(pp));
   dfdx(pp) = a(pp) + 2*b(pp).*abs(x(pp));               %  for |x| <= epsilon  [bigger with xm and decreases with signal (note that b<0)]
   dxdf(pp) = 1./dfdx(pp);
end
if isempty(np)==0
   %f(np) = -(a(np).*abs(x(np))+b(np).*abs(x(np)).^2);      %        for |x| <= epsilon
   x(np) = -(-a(np)+sqrt(a(np).^2-4*b(np).*f(np)))./(2*b(np));
   dfdx(np) = a(np) + 2*b(np).*abs(x(np));   %  for |x| <= epsilon  [bigger with xm and decreases with signal (note that b<0)]
   dxdf(np) = 1./dfdx(np);      
end

x=reshape(x,s);
dxdf=reshape(dxdf,s);

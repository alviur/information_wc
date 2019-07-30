%
%  INTEGRATION OF A SMALL-SCALE WILSON-COWAN MODEL 
% 
%  In this script:
%
%     - I generate the parameters of a small-scale divisive normalization
%       (using parameters_3D_small)  
%
%     - Then, I compute the parameters of Wilson-Cowan (alpha, W) from our relation
%
%     - Note that the parameters of the nonlinearity f are invented (not
%       connected to Div.Norm.)
%       What I am doing is taking the taking the anchor values from a fraction of the
%       average of the Div.Norm. response and using a strong saturation
%       exponent (to moderate attenuation).
%
%     - For each of the considered images (once integrated) I also compute the inverse
%       from the stationary state
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  THINGS TO PLAY WITH:
%
%                     * kappa
%                     * Width of H
%   In Div. Norm.:    * Initial values of c_left, w_right 
%                     * b
%
%   In integration:   * exponent
%                     * k (factor over attenuation) -> affects inverse and will affect H
%                                          
%
%   Intento 1   ------------------------------------------------------------
%                     * csf               [1 0.5 0.3]
%                     * kappa             [1 10 16]
%   In Div. Norm.:    * Width of H        [1 0.06 0.01;0.03 1 0.05;0.001 0.01 1];
%                     * c_left, w_right   [1 2 3]
%                     * b                 0.1*[1 0.6 0.3]  
%
%   In integration:   * exponent          0.4
%                     * k                 [1/4 1/2 1/2].^g
% 
%        MAL: Las marginales inversas preservan peñas
%        MAL: Hay mucha diferencia entre la escala de DN el altas y la de WC
%        MAL: No converge
%
%   Intento 2: reduzco b e incremento kappa para que se incremente DN  --------------------------
%                     * csf               [1 0.5 0.3]
%                     * kappa             [1 10 16]
%   In Div. Norm.:    * Width of H        [1 0.06 0.01;0.03 1 0.05;0.001 0.01 1];
%                     * c_left, w_right   [1 2 3]
%                     * b                 0.1*[1 0.3 0.15]  
%
%   In integration:   * exponent          0.4
%                     * k                 [1/4 1/2 1/2].^g
% 
%        MAL: Las marginales inversas preservan peñas
%        MAL: Hay mucha diferencia entre la escala de DN el altas y la de WC
%        MAL: No converge
%
%   Intento 3: ------------------------
%                     * csf               [1 0.5 0.3]
%                     * kappa             [1 10 16]
%   In Div. Norm.:    * Width of H        [1 0.06 0.01;0.03 1 0.05;0.001 0.01 1];
%                     * c_left, w_right   [1 2 3]
%                     * b                 0.1*[1 0.5 0.4]  
%
%   In integration:   * exponent          0.4
%                     * k                 [1/4 1/2 1/2].^g
% 
%        MAL: Las marginales inversas preservan peñas
%        MAL: Hay mucha diferencia entre la escala de DN el altas y la de WC
%        MAL: No converge
%
%   Intento 4: ------------------------
%                     * csf               [1 0.5 0.3]
%                     * kappa             [1 1 1]
%   In Div. Norm.:    * Width of H        [1 0.06 0.01;0.03 1 0.05;0.001 0.01 1];
%                     * c_left, w_right   [1 2 3]
%                     * b                 0.1*[1 0.5 0.4]  
%
%   In integration:   * exponent          0.4
%                     * k                 [1/4 1/2 1/2].^g
% 
%        BIEN!: Las marginales inversas salen ok (aunque mal escaladas por k)
%        MAL: Hay mas diferencia entre la escala de DN el altas y la de WC (factor 10)
%        BIEN!: converge
%
%   Intento 5: ------------------------
%                     * csf               [1 0.5 0.3]
%                     * kappa             [1 1 1]
%   In Div. Norm.:    * Width of H        [1 0.06 0.01;0.03 1 0.05;0.001 0.01 1];
%                     * c_left, w_right   [1 2 3]
%                     * b                 0.1*[1 0.5 0.4]  
%
%   In integration:   * exponent          0.4
%                     * k                 [1/4 1/14 1/28].^g
% 
%        BIEN!: Las marginales inversas salen ok (aunque mal escaladas por k)
%        MAL: Hay mas diferencia entre la escala de DN el altas y la de WC (factor 10)
%        MAL!: diverge
%
%   Intento 6: ------------------------
%                     * csf               [1 0.5 0.3]
%                     * kappa             [1 1 1]
%   In Div. Norm.:    * Width of H        [1 0.06 0.01;0.03 1 0.05;0.001 0.01 1];
%                     * c_left, w_right   [1 2 3]
%                     * b                 0.1*[1 0.5 0.4]  
%
%   In integration:   * exponent          0.4
%                     * k                 [1/8 1/30 1/60].^g  0.75*k/max(k)
% 
%        BIEN!: Las marginales inversas salen ok
%        MAL: Hay mas diferencia entre la escala de DN el altas y la de WC (factor 10)
%        BIEN!: converge
%

close all
parameters_3D_small  % -> param
close(1),close(2),close(4),close(6),close(7),close(8),close(9),%close(10),close(11)

W = inv(diag(param(2).Hc))*param(2).H*inv(diag(param(2).Hw));
param(2).W = W;
alfa_m = param(2).b./param(2).K;
param(2).alpha = alfa_m;

load images_80  
im = [im1 im2 im3]/256;
x = im2col(im,[1 3],'sliding');
xx=x;

[y1,x1,y2,x2,J1,J2] = stabilized_DN_param(xx,param,0);

xx_a = mean(abs(x2)')';
g = 0.4;
k = diag([1/8 1/15 1/60].^g);
k = 0.75*k/max(k(:));
deltat = 1e-5;
DmW = diag(alfa_m) + W;
Da = diag(alfa_m);
WDxm = W*diag(xx_a.^(1-g));
    
DIF = [];
DIF_x = [];
Xint = [];
X = [];
Y = [];
D = [];
YY = [];
[];

for una_imagen = 1:2:10000;

    % e = abs(y2(:,una_imagen)).^param(2).g;
    e = abs(y2(:,una_imagen)).^g;
    x = abs(x2(:,una_imagen));
    Y = [Y y2(:,una_imagen)];
    X = [X sign(y2(:,una_imagen)).*x];
    
    xt = e;
    
    %% Euler Integration
    %%
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    xs = [];
    %tic
    for t=1:1000
        % Usando la no-linealidad simplificada
        % xtm1 = xt + e*deltat - k*Da*xt*deltat - k*WDxm*(sign(xt).*abs(xt).^g)*deltat;
        
        % Usando la no-linealidad no-singular
        [f,dfdx] = saturation_f(xt,g,xx_a,0.1*xx_a);
        xtm1 = xt + e*deltat - k*Da*xt*deltat - k*W*f*deltat;
        xt = xtm1;
        %  figure(100),plot(xtm1);axis([0 length(e) 0 max(e)]);title(num2str(t))
        xs = [xs xt];
        %if mod(t,500)==0
        %   [una_imagen t]
        %end
    end
    %e_time =toc;  
    if mod(una_imagen,500)==0
       una_imagen
    end    
    dif = xs(:,2:end)-xs(:,1:end-1);
    dif_x = xs - repmat(x,[1 length(xs(1,:))]);
    
    DIF = [DIF;mean(abs(dif).^2)/mean(abs(x).^2)];
    DIF_x = [DIF_x;mean(abs(dif_x).^2)/mean(abs(x).^2)];
    Xint = [Xint sign(y2(:,una_imagen)).*xs(:,end)];
    D = [D xs(:,end)-x]; 
    
    % y = sign(y2(:,una_imagen)).*( k*(Da*xs(:,end) + WDxm*(abs(xs(:,end)).^g)) ).^(1/g);

    [f,dfdx] = saturation_f(xs(:,end),g,xx_a,0.1*xx_a);
    y = sign(y2(:,una_imagen)).*( k*(Da*xs(:,end) + W*f) ).^(1/g);
 
    YY = [YY y];
    
    %figure(1),loglog(mean(abs(dif))/mean(abs(x))),title('updates'),
    %figure(2),loglog(mean(abs(dif_x))./mean(abs(x))),title('Relative MAE from DN'),
    %drawnow
    
end

figure(11),plot3(y1(1,:),y1(2,:),y1(3,:),'.'),title('Luminance')
figure(12),plot3(x1(1,:),x1(2,:),x1(3,:),'.'),title('Brightness')

X1 = param(2).iL*YY;
Y1 = X1.^(1/param(1).g);

figure(13),plot3(Y1(1,:),Y1(2,:),Y1(3,:),'.'),title('Luminance (inverted from WC)')
figure(14),plot3(X1(1,:),X1(2,:),X1(3,:),'.'),title('Brightness (inverted from WC)')

figure(15),plot3(Y(1,:),Y(2,:),Y(3,:),'.'),title('Contrast')
figure(16),plot3(YY(1,:),YY(2,:),YY(3,:),'.'),title('Contrast (inverting WC)')

figure(17),plot3(X(1,:),X(2,:),X(3,:),'.'),title('DN')
figure(18),plot3(Xint(1,:),Xint(2,:),Xint(3,:),'.'),title('WC')


figure(19),loglog(mean(abs(DIF))),title('Relative energy of update'),
figure(20),loglog(mean(abs(DIF_x))),title('Relative MSE from DN'),


figure(21),
subplot(131),plot(Y(1,:),Xint(1,:),'r.'),title('Zero-frequency'),xlabel('Ampl.1 (Brightness)'),ylabel('Response WC')
subplot(132),plot(Y(2,:),Xint(2,:),'g.'),title('Low-frequency'),xlabel('Ampl.2')
subplot(133),plot(Y(3,:),Xint(3,:),'b.'),title('High-frequency'),xlabel('Ampl.3')

figure(210),
subplot(131),plot(YY(1,:),Xint(1,:),'r.'),title('Zero-frequency'),xlabel('Ampl.1 (Brightness)'),ylabel('Response WC')
subplot(132),plot(YY(2,:),Xint(2,:),'g.'),title('Low-frequency'),xlabel('Ampl.2')
subplot(133),plot(YY(3,:),Xint(3,:),'b.'),title('High-frequency'),xlabel('Ampl.3')

figure(211),
subplot(131),plot(Y(1,:),X(1,:),'r.'),title('Zero-frequency'),xlabel('Ampl.1 (Brightness)'),ylabel('Response DN')
subplot(132),plot(Y(2,:),X(2,:),'g.'),title('Low-frequency'),xlabel('Ampl.2')
subplot(133),plot(Y(3,:),X(3,:),'b.'),title('High-frequency'),xlabel('Ampl.3')


[py21,bright2L] = hist(Y(1,:),0.4*nbins);
[py22,c_lowL] = hist(Y(2,:),0.4*nbins);
[py23,c_highL] = hist(Y(3,:),0.4*nbins);
figure(22),semilogy(bright2L,py21,'r-',c_lowL,py22,'g-',c_highL,py23,'b-'),title('PDFs Linear')

[py21i,bright2Li] = hist(YY(1,:),0.4*nbins);
[py22i,c_lowLi] = hist(YY(2,:),0.4*nbins);
[py23i,c_highLi] = hist(YY(3,:),0.4*nbins);
figure(23),semilogy(bright2Li,py21i,'r-',c_lowLi,py22i,'g-',c_highLi,py23i,'b-'),title('PDFs Linear (inverted)')

[px21,bright2NL] = hist(Xint(1,:),0.4*nbins);
[px22,c_lowNL] = hist(Xint(2,:),0.4*nbins);
[px23,c_highNL] = hist(Xint(3,:),0.4*nbins);
figure(24),semilogy(bright2NL,px21,'r-',c_lowNL,px22,'g-',c_highNL,px23,'b-'),title('PDFs WC')

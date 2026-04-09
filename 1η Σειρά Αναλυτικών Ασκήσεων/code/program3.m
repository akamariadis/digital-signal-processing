w = linspace(-2*pi, 2*pi, 4000);
W = abs(mod(w + pi, 2*pi) - pi);
W2 = abs(mod(2*w + pi, 2*pi) - pi); 
H0 = W <= pi/2;
H1 = W > pi/2;
X = W <= 3*pi/4;
X0  = X .* H0;
X0d = 0.5 * ones(size(w));
Q0  = 1 - W/pi;
Z0  = X0d .* Q0;
Z0e = 0.5 * (1 - W2/pi);
Y0  = Z0e .* H0;
X1  = X .* H1;
X1d = 0.5 * (W >= pi/2);
Q1  = W/pi; 
Z1  = X1d .* Q1;
Z1e = 0.5 * (W2 >= pi/2) .* (W2/pi);
Y1  = Z1e .* H1;
Y = Y0 + Y1;

% FIGURE 1 - ΚΛΑΔΟΣ 0
figure('Name', 'Κλάδος k = 0', 'Position', [100, 100, 800, 800]);
subplot(5,1,1); plot(w/pi, X0, 'LineWidth', 1.5); 
title('X_0(e^{j\omega})'); ylabel('Πλάτος'); grid on; ylim([0 1.2]);
subplot(5,1,2); plot(w/pi, X0d, 'LineWidth', 1.5); 
title('X_{0d}(e^{j\omega})'); ylabel('Πλάτος'); grid on; ylim([0 0.8]);
subplot(5,1,3); plot(w/pi, Z0, 'LineWidth', 1.5); 
title('Z_0(e^{j\omega})'); ylabel('Πλάτος'); grid on; ylim([0 0.8]);
subplot(5,1,4); plot(w/pi, Z0e, 'LineWidth', 1.5); 
title('Z_{0e}(e^{j\omega})'); ylabel('Πλάτος'); grid on; ylim([0 0.8]);
subplot(5,1,5); plot(w/pi, Y0, 'LineWidth', 1.5); 
title('Y_0(e^{j\omega})'); xlabel('\omega / \pi'); ylabel('Πλάτος'); grid on; ylim([0 0.8]);

% FIGURE 2 - ΚΛΑΔΟΣ 1
figure('Name', 'Κλάδος k = 1', 'Position', [150, 150, 800, 800]);
subplot(5,1,1); plot(w/pi, X1, 'LineWidth', 1.5, 'Color', 'r'); 
title('X_1(e^{j\omega})'); ylabel('Πλάτος'); grid on; ylim([0 1.2]);
subplot(5,1,2); plot(w/pi, X1d, 'LineWidth', 1.5, 'Color', 'r'); 
title('X_{1d}(e^{j\omega})'); ylabel('Πλάτος'); grid on; ylim([0 0.8]);
subplot(5,1,3); plot(w/pi, Z1, 'LineWidth', 1.5, 'Color', 'r'); 
title('Z_1(e^{j\omega})'); ylabel('Πλάτος'); grid on; ylim([0 0.8]);
subplot(5,1,4); plot(w/pi, Z1e, 'LineWidth', 1.5, 'Color', 'r'); 
title('Z_{1e}(e^{j\omega})'); ylabel('Πλάτος'); grid on; ylim([0 0.8]);
subplot(5,1,5); plot(w/pi, Y1, 'LineWidth', 1.5, 'Color', 'r'); 
title('Y_1(e^{j\omega})'); xlabel('\omega / \pi'); ylabel('Πλάτος'); grid on; ylim([0 1.2]);

% FIGURE 2 - OUTPUT
figure('Name', 'Συνολικό Σύστημα', 'Position', [200, 200, 800, 400]);
subplot(2,1,1);
plot(w/pi, X, 'LineWidth', 2, 'Color', 'k');
title('Αρχικό Σήμα Εισόδου X(e^{j\omega})');
ylabel('Πλάτος'); grid on; ylim([0 1.2]);
subplot(2,1,2);
plot(w/pi, Y, 'LineWidth', 2, 'Color', [0.4940 0.1840 0.5560]);
title('Τελικό Σήμα Εξόδου Y(e^{j\omega}) = Y_0(e^{j\omega}) + Y_1(e^{j\omega})');
xlabel('Συχνότητα (\times \pi rad/sample)'); ylabel('Πλάτος'); grid on; ylim([0 1.2]);
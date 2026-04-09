n12 = 0:11;
n6 = 0:5;
p1 = [1, 2i, 0, 2i, -1, 0, -2, 0, -1, -2i, 1, 0];
p2 = [34, -16, -15, 32, -15, -16];
p3 = [3, -2, -1, 4, -2, 0, 3, -2, -1, 4, -2, 0];
figure('Name', 'Σχεδίαση Σημάτων p1[n], p2[n], p3[n]', 'NumberTitle', 'off');
subplot(2, 2, 1);
stem(n12, real(p1), 'filled', 'b', 'LineWidth', 1.5);
title('Πραγματικό μέρος του p_1[n]');
xlabel('n');
ylabel('Re\{p_1[n]\}');
xlim([-1 12]);
grid on;
%p1[n]
subplot(2, 2, 2);
stem(n12, imag(p1), 'filled', 'r', 'LineWidth', 1.5);
title('Φανταστικό μέρος του p_1[n]');
xlabel('n');
ylabel('Im\{p_1[n]\}');
xlim([-1 12]);
grid on;
%p2[n]
subplot(2, 2, 3);
stem(n6, p2, 'filled', 'k', 'LineWidth', 1.5);
title('Σήμα p_2[n]');
xlabel('n');
ylabel('p_2[n]');
xlim([-1 6]);
grid on;
%p3[n]
subplot(2, 2, 4);
stem(n12, p3, 'filled', 'm', 'LineWidth', 1.5);
title('Σήμα p_3[n]');
xlabel('n');
ylabel('p_3[n]');
xlim([-1 12]);
grid on;
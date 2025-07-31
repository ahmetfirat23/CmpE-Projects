% To comment lines: Ctrl + R
% To uncomment lines: Ctrl + Shift + R

% You can see the values of the variables in the right section "Workspace".
% Double click to see 

% you dont have to put semicolon, but if you don't, result is printed on
% the console:
x = 5
x = 25;

% 5x1 array of 0.0 double
z_1 = zeros([5, 1]);
% 1x5 array of 0.0 double
z_2 = zeros([1, 5]);
% 5x5 array of 0.0 double
z_3 = zeros(5);

r_1 = rand([4, 5]);


u = bitshift(4,3);


x = 5;
% write dbcont to continue
keyboard
disp(x);

% accessing elements and slicing
my_arr = rand([5,5]);
% indices start from 1 !!!
my_arr(1,1)
% slicing: index values can be given as vectors for each dimension
my_arr([1 5 3], 1:2)
% a:b is similar to range(a,b+1) in python
% you can also give step size 
5:0.1:10

% for documentation, you can quickly say
help rand






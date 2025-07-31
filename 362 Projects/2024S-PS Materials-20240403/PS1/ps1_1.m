% To comment lines: Ctrl + R
% To uncomment lines: Ctrl + Shift + R

% You can see the values of the variables in the right section "Workspace".
% Double click to see 

% you dont have to put semicolon, but if you don't, result is printed on
% the console:
x = 5
x = 25;


% numbers are by default of double type
x = 10110;
class(x)
x = 0x10110;
class(x)
x = "hello";
class(x)

% type conversion
x = rand(5)
class(x)
x = uint8(x)
x = int8(x)
x(3,5) = 266

% helpful, common functions
% zeros, ones, rand, randn

% 5x1 array of 
z_1 = zeros([5  1])
% 1x5 array of 
z_2 = ones([1 5])
% 5x5 array of 
z_3 = rand(5)
% any dimension array of
z_4 = randn([1 2 3 4 5])



% similar to print
x = 5;
disp(x);

% accessing elements and slicing
my_arr = rand([5,5])
% indices start from 1 !!!
my_arr(2, 1)
% slicing: index values can be given as vectors for each dimension
% 1:5 is shorthand for creating [1 2 3 4 5]
my_arr([1 5 3], 1:2)
% a:b is similar to range(a,b+1) in python
% you can also give step size 
5:0.1:10


% for documentation, you can quickly say
help rand






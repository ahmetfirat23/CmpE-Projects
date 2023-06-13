`timescale 1ns / 1ns
module source(c, a, b);

input wire [1:0] a; // 2 bit
input wire [1:0] b; // 2 bit
output wire [1:0] c; // 2 bit

// Fill here for the gates
// p'rs + p'qr + pr'
wire n_a1;
wire n_b1;
wire and_n_a1_b1_b0;
wire and_n_a1_a0_b1;
wire and_a1_n_b1;
not(n_a1, a[1]);
not(n_b1, b[1]);
and(and_n_a1_b1_b0, n_a1, b[1], b[0]);
and(and_n_a1_a0_b1, n_a1, a[0], b[1]);
and(and_a1_n_b1, a[1], n_b1);
or(c[1], and_a1_n_b1, and_n_a1_a0_b1, and_n_a1_b1_b0);

// (q+r)(p+q+s)(p'+r)	

wire or_a0b1;
wire or_a1a0b0;
wire or_n_a1_b1;
or(or_a0b1, a[0], b[1]);
or(or_a1a0b0, a[1], a[0], b[0]);
or(or_n_a1_b1, n_a1, b[1]);
and(c[0], or_a1a0b0, or_a0b1, or_n_a1_b1);

endmodule

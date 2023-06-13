`timescale 1ns / 1ns

module encoder(y, x);

// Implement here
input wire[6:0] x;
output reg[2:0] y;
integer i = 0;

always @(x) begin
    y = 0;
    for (i = 0; i < 7; i = i + 1) begin
        y = y + x[i];
    end
end



endmodule

module mux(z, y, s);

// Implement here
input wire[2:0] y;
input wire[1:0] s;
output reg z;

always @(y,s) begin

    if(y[0] + y[1] + y[2] == s) begin
        z = 1;
    end
    else begin
        z = 0;
    end
    
end


endmodule
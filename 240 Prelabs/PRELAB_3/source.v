`timescale 1ns / 1ns

// Put the last sequence you have tried below.
// parameter inputseq = 64'b1101001000100011000001110000100001100000111100001000100011010101;

module source(y, stateReg, nextStateReg, x, rst, clk);

// Fill in here
output reg[0:0]y;
input wire x;
input wire rst;
input wire clk;
parameter   S0 = 3'b000,
            S1 = 3'b001,
            S2 = 3'b010,
            S3 = 3'b011,
            S4 = 3'b100,
            S5 = 3'b101,
            S6 = 3'b110;

output reg[2:0] stateReg;
output reg[2:0] nextStateReg;

always@(stateReg, x) begin
    case(stateReg)
        S0: begin
            y <= 1'b0;
            if(x == 0) begin
                nextStateReg <= S1;
            end
            else begin
                nextStateReg <= S4;
            end
        end
        S1: begin
            y <= 1'b0;
            if(x == 0) begin
                nextStateReg <= S2;
            end
            else begin
                nextStateReg <= S4;
            end
        end
        S2: begin
            y <= 1'b0;
            if(x == 0) begin
                nextStateReg <= S2;
            end
            else begin
                nextStateReg <= S3;
            end
        end
        S3: begin
            y <= 1'b1;
            if(x == 0) begin
                nextStateReg <= S1;
            end
            else begin
                nextStateReg <= S5;
            end
        end
        S4: begin
            y <= 1'b0;
            if(x == 0) begin
                nextStateReg <= S1;
            end
            else begin
                nextStateReg <= S5;
            end
        end
        S5: begin
            y <= 1'b0;
            if(x == 0) begin
                nextStateReg <= S6;
            end
            else begin
                nextStateReg <= S5;
            end
        end
        S6: begin
            y <= 1'b1;
            if(x == 0) begin
                nextStateReg <= S2;
            end
            else begin
                nextStateReg <= S4;
            end
        end
    endcase
end
always@(posedge clk) begin
	if(rst) begin
		stateReg <= S0;
	end
	else begin
		stateReg <= nextStateReg;
	end
end


endmodule

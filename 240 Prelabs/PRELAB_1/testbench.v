`timescale 1ns / 1ns
module testbench();

reg [1:0] aa;
reg [1:0] bb;
wire [1:0] cc;

source s0(cc, aa, bb);

initial begin
    $dumpfile("TimingDiagram.vcd");
    $dumpvars(0, cc, aa, bb);
    	
	aa = 2'b00;
	bb = 2'b00;
	#20;
	aa = 2'b00;
	bb = 2'b01;
	#20;
	aa = 2'b00;
	bb = 2'b10;
	#20;
	aa = 2'b00;
	bb = 2'b11;
	#20;
	
	aa = 2'b01;
	bb = 2'b00;
	#20;
	aa = 2'b01;
	bb = 2'b01;
	#20;
	aa = 2'b01;
	bb = 2'b10;
	#20;
	aa = 2'b01;
	bb = 2'b11;
	#20;
	
	aa = 2'b10;
	bb = 2'b00;
	#20;
	aa = 2'b10;
	bb = 2'b01;
	#20;
	aa = 2'b10;
	bb = 2'b10;
	#20;
	aa = 2'b10;
	bb = 2'b11;
	#20;
	
	aa = 2'b11;
	bb = 2'b00;
	#20;
	aa = 2'b11;
	bb = 2'b01;
	#20;
	aa = 2'b11;
	bb = 2'b10;
	#20;
	aa = 2'b11;
	bb = 2'b11;
	#20;
	
    $finish;
end

endmodule
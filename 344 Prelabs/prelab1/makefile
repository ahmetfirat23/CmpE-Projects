debug:	output.elf
		whisper --interactive output.elf
output.elf:
		riscv32-unknown-elf-gcc *.s -o output.elf -nostdlib
		riscv32-unknown-elf-objdump --disassemble output.elf > output.dis
clear:
		rm -rf *.elf *.dis
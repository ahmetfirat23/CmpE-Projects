#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
// constants
int NUM_INSTRUCTIONS = 21;
int MAX_PROCESSES = 10;
int context_switch_time = 10;
int silver_quantum = 80;
int gold_quantum = 120;
int platinum_quantum = 120;
int thresh_to_gold = 3;
int thresh_to_plat = 5;

// process types
typedef enum {
    PLATINUM,
    GOLD,
    SILVER,
} ProcessType;

// process names
typedef enum {
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9,
    P10,
} ProcessName;

// helper functions
ProcessType get_process_type(char* name) {
    if (strcmp(name, "PLATINUM") == 0) {
        return PLATINUM;
    } else if (strcmp(name, "GOLD") == 0) {
        return GOLD;
    } else {
        return SILVER;
    }
}

ProcessName get_process_name(char* name){
    if (strcmp(name, "P1") == 0) {
        return P1;
    } else if (strcmp(name, "P2") == 0) {
        return P2;
    } else if (strcmp(name, "P3") == 0) {
        return P3;
    } else if (strcmp(name, "P4") == 0) {
        return P4;
    } else if (strcmp(name, "P5") == 0) {
        return P5;
    } else if (strcmp(name, "P6") == 0) {
        return P6;
    } else if (strcmp(name, "P7") == 0) {
        return P7;
    } else if (strcmp(name, "P8") == 0) {
        return P8;
    } else if (strcmp(name, "P9") == 0) {
        return P9;
    } else {
        return P10;
    }
}

// structs
typedef struct Instruction{
    char name[10];
    int duration;
} Instruction;

typedef struct Process{
    ProcessName name;
    int priority;
    int arrival_time;
    int abs_arrival_time;
    ProcessType type;

    Instruction *instructions;
    int num_instructions;
    int last_executed_instruction;

    int quantum;
    int quantum_count; // number of quantums that the process executed in type 
    int cpu_time;

    int waiting_time;
    int turnaround_time;
} Process;

// promotion functions
// changes type, limit and resets quantum count 
void promote_to_gold(Process *process) {
    // printf("===========================\nPromoting process %d to gold\n===========================\n", process->name+1);
    process->type = GOLD;
    process->quantum = gold_quantum;
    process->quantum_count = 0;
}

void promote_to_plat(Process *process) {
    // printf("===========================\nPromoting process %d to platinum\n===========================\n", process->name+1);
    process->type = PLATINUM;
    process->quantum = platinum_quantum;
    process->quantum_count = 0;
}

// executes the next instruction in the process
int execute_process(Process *process, int current_time) {
    process->last_executed_instruction++;
    int inst_idx = process->last_executed_instruction;
    Instruction *instruction = &process->instructions[inst_idx];
    // printf("Executing Instruction %s for Process %d for %d duration at start time %d ending time: %d\n", instruction->name, process->name+1, instruction->duration, current_time, current_time+instruction->duration);
    process->cpu_time += instruction->duration;

    return instruction->duration;
}

// 1 if A has priority, -1 if B has priority, 0 if equal
// comparator for priority queue
int compare(Process A, Process B, int current_time){
    // check if processes are finished
    if (A.last_executed_instruction == A.num_instructions-1 && B.last_executed_instruction != B.num_instructions-1){
        return -1;
    }
    else if (A.last_executed_instruction != A.num_instructions-1 && B.last_executed_instruction == B.num_instructions-1){
        return 1;
    }
    else {
        // check if processes arrived
        if (A.arrival_time > current_time && B.arrival_time <= current_time){
            return -1;
        }
        else if (B.arrival_time > current_time){
            return 1;
        }
        else{
            // check if processes are platinum
            if (A.type == PLATINUM && B.type != PLATINUM){
                return 1;
            }
            else if (A.type != PLATINUM && B.type==PLATINUM){
                return -1;
            }
            else{
                // check for priority
                if (A.priority > B.priority){
                    return 1;
                }
                else if (A.priority < B.priority){
                    return -1;
                }
                else{
                    // check for arrival time
                    if (A.arrival_time < B.arrival_time){
                        return 1;
                    }
                    else if (A.arrival_time > B.arrival_time){
                        return -1;
                    }
                    else{
                        // check for names
                        if (A.name < B.name){
                            return 1;
                        }
                        else{
                            return -1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

// sorts processes by priority, arrival time, type, and name
void bubble_sort(Process *processes, int num_processes, int current_time){
    for (int i = 0; i < num_processes-1; i++){
        for (int j = i + 1; j < num_processes; j++){
            int comparison = compare(processes[i], processes[j], current_time);
            if (comparison == -1){
                Process temp = processes[i];
                processes[i] = processes[j];
                processes[j] = temp;
            }
        }
    }
}

// checks if all processes are complete
int check_all_complete(Process *processes, int num_processes){
    for (int i = 0; i < num_processes; i++){
        if (processes[i].last_executed_instruction < processes[i].num_instructions-1){
            return 0;
        }
    }
    return 1;
}

// if there is no process to run, idle the cpu until the next process arrives
int idle_cpu(Process *processes, int num_processes, int current_time){
    int min_arrival_time = processes[0].arrival_time;
    for (int i = 0; i < num_processes; i++){
        if (processes[i].last_executed_instruction != processes[i].num_instructions - 1 && processes[i].arrival_time < min_arrival_time){
            min_arrival_time = processes[i].arrival_time;
        }
    }
    if (min_arrival_time > current_time){
        current_time = min_arrival_time;
    }
    // else{
    //     printf("#####\nThere is something wrong!!\n####\n");
    // }
    bubble_sort(processes, num_processes, current_time);
    return current_time;
}

// returns a reference to the process in the array
Process* get_process_reference(Process process, Process *processes, int num_processes){
    for (int i = 0; i < num_processes; i++){
        if (processes[i].name == process.name){
            return &processes[i];
        }
    }
    return NULL;
}

// prints all processes for debugging
void print_all_processes(int num_processes, Process *processes)
{
    for (int i = 0; i < num_processes; i++)
    {
        printf("%d %d %d %d\n", processes[i].name, processes[i].priority, processes[i].arrival_time, processes[i].type);
        for (int j = 0; j < processes[i].num_instructions; j++)
        {
            printf("%s %d\n", processes[i].instructions[j].name, processes[i].instructions[j].duration);
        }
    }
}

// main function
int main() {
    // read input file
    char def_file[20] = "definition.txt";
    char instruction_file[20] = "instructions.txt";
    char buffer[50];
    FILE *fp = fopen(instruction_file, "r");

    // form instructions array
    Instruction *instructions = malloc(sizeof(Instruction) * NUM_INSTRUCTIONS);
    int num_instructions = 1;
    while (fgets(buffer, 50, fp) != NULL) {
        Instruction *instruction = malloc(sizeof(Instruction));
        char name[10];
        sscanf(buffer, "%s %d", instruction->name, &instruction->duration);
        instructions[num_instructions] = *instruction;
        num_instructions++;
        if (num_instructions == 21)
            num_instructions = 0;
    }
    fclose(fp);

    // form processes array
    fp = fopen(def_file, "r");
    Process *processes = malloc(sizeof(Process) * MAX_PROCESSES);
    int num_processes = 0;
    while (fgets(buffer, 50, fp) != NULL) {
        Process *process = malloc(sizeof(Process));
        char name[10];
        char type[10];
        sscanf(buffer, "%s %d %d %s", name, &process->priority, &process->arrival_time, type);
        process->abs_arrival_time = process->arrival_time;
        process->name = get_process_name(name);
        process->type = get_process_type(type);
        process->quantum_count = 0;
        process->cpu_time = 0;
        process->waiting_time = 0;
        process->turnaround_time = 0;
        process->last_executed_instruction = -1; // -1 means no instruction has been executed yet
        if(process->type == PLATINUM) {
            process->quantum = platinum_quantum;
        } else if (process->type == GOLD) {
            process->quantum = gold_quantum;
        } else if (process->type == SILVER) {
            process->quantum = silver_quantum;
        }
        // read instructions of the process
        strcat(name, ".txt");
        FILE *fp2 = fopen(name, "r");
        int num_instructions = 0;
        process->instructions = malloc(sizeof(Instruction) * 15);
        while (fgets(buffer, 50, fp2) != NULL) {
            char* digit_start = buffer + strcspn(buffer, "123456789");
            char name[10];
            int index =  (int)strtol(digit_start, NULL, 10); 
            process->instructions[num_instructions] = instructions[index];
            num_instructions++;
        }
        fclose(fp2);
        process->num_instructions = num_instructions;
        processes[num_processes] = *process;
        num_processes++;
    }
    fclose(fp);

    // start the scheduler
    int current_time = 0;
    bubble_sort(processes, num_processes, current_time);
    // get the first process in array (highest priority)
    Process* current_process = &processes[0];
    Process prev_process;
    // do context switch before starting
    current_time += context_switch_time;

    // run until all processes are complete
    while (check_all_complete(processes, num_processes) < 1){

        // execute the process
        current_time += execute_process(current_process, current_time);

        // check if process is completed
        if (current_process->last_executed_instruction == current_process->num_instructions-1){
            // record waiting time and turnaround time
            current_process->turnaround_time = current_time - current_process->abs_arrival_time;
            int burst_time = 0;
            for (int i = 0; i < current_process->num_instructions; i++){
                burst_time += current_process->instructions[i].duration;
            }
            current_process->waiting_time = current_process->turnaround_time - burst_time;
            current_process->type = 10;
            // printf("############################################\n");
            // printf("Process %d finished at time %d\n", current_process->name+1, current_time);
            // printf("############################################\n");
        }
        if (current_process->type != PLATINUM){
            // process exceed its quantum
            // run self or round robin or highest priority
            if (current_process->cpu_time >= current_process->quantum){
                current_process->cpu_time = 0;
                current_process->quantum_count++;
                // update the arrival time of reenterence time to queue
                current_process->arrival_time = current_time;
                if (current_process->type==GOLD && current_process->quantum_count >= thresh_to_plat){
                    promote_to_plat(current_process);
                }
                else if (current_process->type==SILVER && current_process->quantum_count >= thresh_to_gold){
                    promote_to_gold(current_process);
                }

                // bubble sort modifies current process, so save it
                prev_process = *current_process;
                bubble_sort(processes, num_processes, current_time);
                // if highest priority process hasn't arrive then no other process arrived
                if (processes[0].arrival_time > current_time){
                    current_time = idle_cpu(processes, num_processes, current_time);
                }
                // if a new process has the highest priority, context switch
                if (processes[0].name != prev_process.name){
                    current_time += context_switch_time;
                }                
                current_process = &processes[0];
            }     
            // process does not exceed its quantum
            // run self or highest priority  
            else {
                // bubble sort modifies current process, so save it
                prev_process = *current_process;
                bubble_sort(processes, num_processes, current_time);
                // get the reference to the current process in the new array
                current_process = get_process_reference(prev_process, processes, num_processes);
                // if highest priority process hasn't arrive then no other process arrived
                if (processes[0].arrival_time > current_time){
                    current_time = idle_cpu(processes, num_processes, current_time);
                }
                // if a new process has the highest priority, context switch
                if (prev_process.name != processes[0].name){
                    current_process->quantum_count += 1;
                    current_process->cpu_time = 0;
                    current_process->arrival_time = current_time;
                    if (current_process->type==GOLD && current_process->quantum_count >= thresh_to_plat){
                        promote_to_plat(current_process);
                    }
                    else if (current_process->type==SILVER &&             current_process->quantum_count >= thresh_to_gold){
                        promote_to_gold(current_process);
                    }
                    current_time += context_switch_time;
                }
                current_process = &processes[0];     
            }
        }      
    }

    // for (int i = 0; i < num_processes; i++){
    //     printf("Process %d waiting time: %d\n", processes[i].name+1, processes[i].waiting_time);
    //     printf("Process %d turnaround time: %d\n", processes[i].name+1, processes[i].turnaround_time);
    // }
    float average_waiting_time = 0;
    float average_turnaround_time = 0;
    for (int i = 0; i < num_processes; i++){
        average_waiting_time += processes[i].waiting_time;
        average_turnaround_time += processes[i].turnaround_time;
    }
    average_waiting_time = average_waiting_time/num_processes;
    average_turnaround_time = average_turnaround_time/num_processes;
    // if float, print with one decimal place
    if (average_waiting_time - (int)average_waiting_time != 0){
        printf("%.1f\n", average_waiting_time);
    }
    else{
        int int_average_waiting_time = (int)average_waiting_time;
        printf("%d\n", int_average_waiting_time);
    }
    if (average_turnaround_time - (int)average_turnaround_time != 0){
        printf("%.1f\n", average_turnaround_time);
    }
    else{
        int int_average_turnaround_time = (int)average_turnaround_time;
        printf("%d\n", int_average_turnaround_time);
    }
    return 0;
}


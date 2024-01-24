#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <fcntl.h>
#include <pwd.h>
#include <time.h>
#include <sys/sysctl.h>
//  Limit on user input size, path address size, info text sizes and buffered output sizes (multiplied by 5)
#define MAX_INPUT_SIZE 1024
// Limit on number of arguments
#define MAX_ARG_SIZE 128
// Aliases will be saved to this file
#define ALIAS_FILE "aliases.txt"

struct Alias {
    char name[MAX_INPUT_SIZE];
    char* args[MAX_INPUT_SIZE];
};

struct Alias* aliases;
int alias_count = 0;
char last_executed[MAX_ARG_SIZE];
char last_candidate[MAX_ARG_SIZE];

void execute_command(char** args, char* file_name, int background, int append, int reversed);
int redirect_output(char *file_name, int append);
void print_initial();
void find_from_path(char* command, char** args, char* file_name, int background, int append, int reversed);
void save_alias();
void bello(char* file_name, int append, int reversed);

// Replicates zsh behavior on ctrl+c
void sigint_handler(int sig){
    signal(SIGINT, sigint_handler);
    printf("\n");
    print_initial();
    fflush(stdout);
}

// When a foreground process is running, ctrl+c should not kill the shell
// Also initials shouldn't be printed twice
void fore_signint_handler(int sig){
    signal(SIGINT, fore_signint_handler);
    printf("\n");
    fflush(stdout);
}

// These functions just for fun
void sigquit_handler(int sig){
    signal(SIGQUIT, sigquit_handler);
    printf("Quiting on SIGQUIT signal...\n");
    fflush(stdout);
    save_alias();
    exit(1);
}
void sigterm_handler(int sig){
    signal(SIGTERM, sigterm_handler);
    printf("Quiting on SIGTERM signal...\n");
    fflush(stdout);
    save_alias();
    exit(0);
}

// Collects zombie processes as soon as they terminate and prints their pid with exit status
void sigchild_handler(int sig){
    signal(SIGCHLD, sigchild_handler);
    int status;
    pid_t pid;
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        printf("[%d] exited %d\n", pid, status);
        fflush(stdout);
    }

    print_initial();
    fflush(stdout);
}

// Creates alias if it doesn't exist, updates if it does
void create_alias(char * args[MAX_ARG_SIZE]){
    // Reallocate memory if needed
    if (alias_count >= MAX_INPUT_SIZE) {
        aliases = (struct Alias*)realloc(aliases, sizeof(struct Alias) * (alias_count + MAX_INPUT_SIZE));
    }

    // Check if alias already exists
    int alias_index = alias_count;
    for(int i = 0; i < alias_count; i++){
        if (strcmp(aliases[i].name, args[1]) == 0){
            alias_index = i;
            break;
        }
    }
    strcpy(aliases[alias_index].name, args[1]);

    int j = 3;
    int padding = 3;
    while(args[j]!=NULL){       
        aliases[alias_index].args[j-padding] = strdup(args[j]);
        j++;
    }
    aliases[alias_index].args[j-padding] = NULL;

    if (alias_index == alias_count){
        alias_count++;
    }

    // Save aliases to file after each change
    save_alias();
}

// Saves aliases to file
void save_alias(){
    FILE *file = fopen(ALIAS_FILE, "w");
    if (file == NULL) {
        perror("Error opening aliases file");
        return;
    }

    for (int i = 0; i < alias_count; i++){
        fprintf(file, "%s =", aliases[i].name);
        for (int j = 0; j < MAX_ARG_SIZE; j++){
            if (aliases[i].args[j] == NULL){
                break;
            }
            fprintf(file, " %s", aliases[i].args[j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// Loads aliases from file
void load_aliases(){
    FILE *file = fopen(ALIAS_FILE, "r");
    // No aliases file exists
    if (file == NULL) {
        return; 
    }

    char line[MAX_INPUT_SIZE];

    while (fgets(line, sizeof(line), file) != NULL) {
        // Simple parsing
        line[strcspn(line, "\n")] = '\0';
        char* args[MAX_ARG_SIZE];
        args[0] = "alias";
        char* token = strtok(line, " ");
        int i = 1;
        while (token != NULL) {
            args[i++] = token;
            token = strtok(NULL, " ");
        }
        args[i] = NULL;
        create_alias(args);
    }
    fclose(file);
}

// Finds the command in aliases and directs to find_from_path
int find_from_alias(char* command, char** args, char* file_name, int background, int append, int reversed){
    for(int i = 0; i < alias_count; i++){       
        if (strcmp(aliases[i].name, command) == 0){
            char* new_args[MAX_ARG_SIZE];
            int j = 0;
            // Concatenate saved command and user entered arguments
            while(aliases[i].args[j] != NULL){
                new_args[j] = strdup(aliases[i].args[j]);
                j++;
                if (aliases[i].args[j] == NULL){
                    int k = 1;
                    while(args[k] != NULL){
                        new_args[j++] = strdup(args[k++]);
                    }
                    new_args[j] = NULL;
                }
            }

            // Parse background and redirection operators
            int l = 1;
            int parsing_error = 0;
            while(new_args[l] != NULL){
                char* token = new_args[l];
                if (strcmp(token, "&")== 0){
                    new_args[l] = NULL;
                    if (new_args[l+1] != NULL){
                        fprintf(stderr, "Parsing error after &!\n");
                        parsing_error = 1;
                        break;
                    }
                    background = 1;
                }
                else if (strcmp(token, ">")== 0){
                    new_args[l] = NULL;
                    file_name = new_args[l+1];
                    if (file_name == NULL){
                        fprintf(stderr, "Parsing error after >!\n");
                        parsing_error = 1;
                        break;
                    }
                }
                else if (strcmp(token, ">>")== 0){
                    new_args[l] = NULL;
                    file_name = new_args[l+1];
                    if (file_name == NULL){
                        fprintf(stderr, "Parsing error after >>!\n");
                        parsing_error = 1;
                        break;
                    }
                    append = 1;
                }
                else if (strcmp(token, ">>>")== 0){
                    new_args[l] = NULL;
                    file_name = new_args[l+1];
                    if (file_name == NULL){
                        fprintf(stderr, "Parsing error after >>>!\n");
                        parsing_error = 1;
                        break;
                    }
                    append = 1;
                    reversed = 1;
                }
                l++;
            }

            // If parsing error, don't execute command
            if (parsing_error){
                return parsing_error;
            }
            // Check for built-in commands 
            // please don't use nested alias or exit and bello as an alias name
            if (strcmp(new_args[0], "exit") == 0){
                save_alias();
                exit(0);
            }
            else if (strcmp(new_args[0], "bello")== 0){
                bello(file_name, append, reversed);
                return 0;
            }

            // Send formatted args list to path search
            find_from_path(new_args[0], new_args, file_name, background, append, reversed);
            return 0;
        }
    }

    // Alias not found
    return -1;
}

// Finds the command in path and directs to execute_command
void find_from_path(char* command, char** args, char* file_name, int background, int append, int reversed){
    char* path = getenv("PATH");
    // can't use string literal with strtok it gives segmentation error
    char* path_copy = strdup(path); 
    char* address = strtok(path_copy, ":");
    while(address != NULL){
        // need an array to concatenate the address and command
        char command_address[MAX_INPUT_SIZE]; 
        // really cool function for concatenation
        snprintf(command_address, sizeof(command_address), "%s/%s", address, command); 
        // checks the file path is executable
        if (access(command_address, X_OK) ==0){ 
            // release this becase strdup uses malloc
            free(path_copy); 
            args[0] = command_address;
            // Send for execution
            execute_command(args, file_name, background, append, reversed);
            return;
        }
        address = strtok(NULL, ":");
    }
    fprintf(stderr, "Command not found: %s\n", command);
    free(path_copy);
}

// Executes command if re-redirection and background operator given at same time
// Forks a child and a grandchild, child reads grandchild output from fork and 
// prints it to file in reverse order
void reversed_background(char** args, char* file_name){    
    pid_t child_pid = fork();

    if (child_pid == -1){
        perror("Forking failed!");
    }
    // Parent process
    else if (child_pid > 0){
        printf("[%d] reverse handler process started\n", child_pid);
        printf("[%d] started\n", child_pid + 1);
        // Set last executed here since we can't know grandchild's future exit status
        last_executed[0] = '\0';
        strcpy(last_executed, last_candidate);
        return;
    }
    // Child process
    else{
        // Use pipe to read from grandchild
        int pipe_fd[2];
        pipe(pipe_fd);

        pid_t grandchild_pid = fork();
        
        if (grandchild_pid == -1){
            perror("Forking failed!");
            exit(1);
        }
        // Child process
        else if (grandchild_pid > 0){
            close(pipe_fd[1]);

            int status;
            waitpid(grandchild_pid, &status, 0);
            
            if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
                exit(1);
            }
            else{
                // read from pipe and write to file
                int fd = open(file_name, O_WRONLY | O_CREAT | O_APPEND, 0644);
                // If file couldn't be opened, don't execute command
                if (fd == -1){
                    perror("Error opening file!");
                    close(pipe_fd[0]);
                    exit(1);
                }
                char buffer[MAX_INPUT_SIZE * 5];
                ssize_t bytes_read = read(pipe_fd[0], buffer, sizeof(buffer));
                for (ssize_t i = bytes_read - 1; i >= 0; i--){
                    write(fd, &buffer[i], 1);
                }
                close(pipe_fd[0]);
            }
            // Exit child or else it will continue to execute a parallel myshell
            exit(0);
        }
        // Grandchild process
        else{
            // close read end of pipe
            close(pipe_fd[0]);
            // direct stdout to pipe          
            dup2(pipe_fd[1], STDOUT_FILENO);   
            // close write end of pipe
            close(pipe_fd[1]);      
            execv(args[0], args);
            // I don't expect this to be ever executed
            perror("Command couldn't be executed!");          
        }
    }

}

// Executes given command
void execute_command(char** args, char* file_name, int background, int append, int reversed){
    // Special case
    if (reversed && background){
        reversed_background(args, file_name);
        return;
    }
    // Save stdout to restore later
    int stdout_backup = dup(fileno(stdout));

    // Create pipe for reversed redirection
    int pipe_fd[2];
    if (reversed){
        pipe(pipe_fd);
    }

    pid_t pid = fork();
    // Forking error
    if (pid == -1){ 
        // errno is set here so perror used
        perror("Forking failed!");        
    }
    // Parent process
    else if (pid > 0){ 
        if (reversed){
            // close write end of pipe
            close(pipe_fd[1]);
        }

        if (background==0){
            // Set signal handlers for foreground process
            signal(SIGINT, fore_signint_handler);
            signal(SIGCHLD, SIG_DFL);

            int status;
            waitpid(pid, &status, 0);
            // Set signal handlers back to default
            signal(SIGINT, sigint_handler);
            signal(SIGCHLD, sigchild_handler);

            // If child exited with success, update last executed command
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                last_executed[0] = '\0';
                strcpy(last_executed, last_candidate);

                if (reversed){
                    // read from pipe and write to file
                    int fd = open(file_name, O_WRONLY | O_CREAT | O_APPEND, 0644);
                    // If file couldn't be opened, don't output
                    if (fd == -1){
                        perror("Error opening file!");
                        close(pipe_fd[0]);
                        return;
                    }
                    char buffer[MAX_INPUT_SIZE * 5];
                    ssize_t bytes_read = read(pipe_fd[0], buffer, sizeof(buffer));
                    for (ssize_t i = bytes_read - 1; i >= 0; i--){
                        write(fd, &buffer[i], 1);
                    }
                    close(fd);
                    close(pipe_fd[0]);
                }   
            } 
        }
        // Background process
        else{
            // zsh also prints this type of message
            printf("[%d] started\n", pid);
            // Set last executed here since we can't know child's future exit status
            last_executed[0] = '\0';
            strcpy(last_executed, last_candidate);
        }      
    }

    // Child process
    else{ 
        // Set signal handlers back to default so it doesn't print pid
        signal(SIGINT, SIG_DFL);
        if (reversed){
            // close read end of pipe
            close(pipe_fd[0]);
            // direct stdout to pipe
            dup2(pipe_fd[1], STDOUT_FILENO); 
            close(pipe_fd[1]);
        }
        // This is for > and >> operators
        else if (file_name != NULL){
            int redirect_no = redirect_output(file_name, append);
            // If file couldn't be opened, don't execute command
            if (redirect_no == -1){
                return;
            }
        }
        // Execute command
        execv(args[0], args);
        // I don't expect this to be ever executed
        perror("Command couldn't be executed!");
        dup2(stdout_backup, STDOUT_FILENO);
    }
}

// Redirects stdout to given file
int redirect_output(char* file_name, int append){
    int fd;
    // If append is true, append to file, else truncate
    // 0644 is the permission zsh uses
    if (append){
        fd = open(file_name, O_WRONLY | O_CREAT | O_APPEND, 0644);
    }
    else{
        fd = open(file_name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    }
    if (fd == -1){
        perror("Error opening file!");
        return -1;
    }
    // direct stdout to file opened
    dup2(fd, STDOUT_FILENO); 
    close(fd);
    return 0;
}

// Removes quotes from given string
void remove_quotes(char *str) {
    int len = strlen(str);
    int current_position = 0;
    for (int i = 0; i < len; ++i) {
        if (str[i] != '"') {
            str[current_position] = str[i];
            current_position++;
        }
    }
    str[current_position] = '\0';
}

// Prints bello information
void bello(char* file_name, int append, int reversed) {
    // Nothing fancy here, just function calls
    char username[MAX_INPUT_SIZE];
    char hostname[MAX_INPUT_SIZE];
    getlogin_r(username, sizeof(username));
    gethostname(hostname, sizeof(hostname));

    char *tty = ttyname(STDIN_FILENO);
    char *shell_name = getenv("SHELL");

    struct passwd *pw = getpwuid(getuid());
    char *homedir = pw->pw_dir;

    time_t current_time;
    struct tm *time_info;
    time(&current_time);
    time_info = localtime(&current_time);
    char time_str[80];
    // Year-Month-Day Hour:Minute:Second
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", time_info);

    // This section runs ps command to get number of processes
    // Used ps command and parsed its output
    // Subtract 5 from the output
    // 5 is the number of processes in current tty when there is no background processes
    // Main myshell process is not counted
    char output[64];
    char system_call[64];
    int stdout_backup = dup(fileno(stdout));
    int pipe_fd[2];
    pipe(pipe_fd);
    dup2(pipe_fd[1], fileno(stdout));
    close(pipe_fd[1]);

    sprintf(system_call, "ps -t %s -o tty= | wc -l", tty);
    system(system_call);

    dup2(stdout_backup, fileno(stdout));
    ssize_t read_size = read(pipe_fd[0], output, sizeof(output));
    int count;
    sscanf(output, "%d", &count);
    close(pipe_fd[0]);
    count = count - 5;

    if (file_name != NULL){
        int redirect_no = redirect_output(file_name, append);
        if (redirect_no == -1){
            return;
        }
    }

    // If reversed store in a buffer and write to file in reverse order
    if (reversed){
        int fd;
        fd = open(file_name, O_WRONLY | O_CREAT | O_APPEND, 0644);   
        char buffer[MAX_INPUT_SIZE * 5];
        sprintf(buffer, "Username: %s\n", username);
        sprintf(buffer, "%sHostname: %s\n", buffer, hostname);
        sprintf(buffer, "%sLast Executed Command: %s\n", buffer, last_executed);
        sprintf(buffer, "%sTTY: %s\n", buffer, tty);
        sprintf(buffer, "%sCurrent Shell Name: %s\n", buffer, shell_name);
        sprintf(buffer, "%sHome Location: %s\n", buffer, homedir);
        sprintf(buffer, "%sCurrent Time and Date: %s\n", buffer, time_str);
        sprintf(buffer, "%sCurrent Number of Processes: %d\n", buffer, count);
        ssize_t bytes_read = strlen(buffer);
        for (ssize_t i = bytes_read - 1; i >= 0; i--){
            write(fd, &buffer[i], 1);
        }
        close(fd);
    }
    else{
        // Print the information
        printf("Username: %s\n", username);
        printf("Hostname: %s\n", hostname);
        printf("Last Executed Command: %s\n", last_executed);
        printf("TTY: %s\n", tty);
        printf("Current Shell Name: %s\n", shell_name);
        printf("Home Location: %s\n", homedir);
        printf("Current Time and Date: %s\n", time_str);
        printf("Current Number of Processes: %d\n", count);
    }
    dup2(stdout_backup, STDOUT_FILENO);
}

// Prints initial prompt
void print_initial()
{
    char username[MAX_INPUT_SIZE];
    char hostname[MAX_INPUT_SIZE];
    char current_path[MAX_INPUT_SIZE];
    getlogin_r(username, sizeof(username));
    gethostname(hostname, sizeof(hostname));
    getcwd(current_path, sizeof(current_path));
    printf("%s@%s %s --- ", username, hostname, current_path);
}

// Main function
int main(){
    // Signal handlers
    signal(SIGINT, sigint_handler);
    signal(SIGQUIT, sigquit_handler);
    signal(SIGTERM, sigterm_handler); 
    signal(SIGCHLD, sigchild_handler);
    
    // Allocate memory for aliases
    aliases = (struct Alias*)malloc(sizeof(struct Alias) * MAX_INPUT_SIZE);
    load_aliases();

    char input[MAX_INPUT_SIZE];
    char command[MAX_INPUT_SIZE];

    while (true){
        // Reset flags and variables
        int append = 0;
        int reversed = 0;
        int background = 0;
        int alias_create = 0;
        int alias_no = -1;
        int parsing_error = 0;
        char* file_name = NULL;
        
        print_initial();
        if (fgets(input, sizeof(input), stdin) == NULL) {
            // EOF
            fprintf(stderr, "Error on reading input stream!\n");
            save_alias();
            exit(1);
        }

        // Remove trailing newline character
        input[strcspn(input, "\n")] = '\0'; 
        
        // Save candidate user input for last executed command
        strcpy(last_candidate, input);
        
        remove_quotes(input);
        // Get the first token (command)
        char* command = strtok(input, " "); 

        // continue on empty input
        if (command == NULL){
            continue;
        }
        else if (strcmp(command, "alias") == 0){
            alias_create = 1;
        }

        char* args[MAX_ARG_SIZE];
        args[0] = command;
        char* token = strtok(NULL, " ");

        int i = 1;
        while (token != NULL){
            if (alias_create == 0){
                if (strcmp(token, "&")== 0){
                    // & operator must be the last token
                    if (strtok(NULL, " ") != NULL){
                        fprintf(stderr, "Parsing error after &!\n");
                        parsing_error = 1;
                        break;
                    }
                    background = 1;
                }

                if (strcmp(token, ">")== 0){
                    file_name = strtok(NULL, " ");
                    // > operator must be followed by a file name
                    if (file_name == NULL){
                        fprintf(stderr, "Parsing error after >!\n");
                        parsing_error = 1;
                        break;
                    }
                }
                else if (strcmp(token, ">>")== 0){
                    file_name = strtok(NULL, " ");
                    // >> operator must be followed by a file name
                    if (file_name == NULL){
                        fprintf(stderr, "Parsing error after >>!\n");
                        parsing_error = 1;
                        break;
                    }
                    append = 1;
                }
                else if (strcmp(token, ">>>")== 0){
                    file_name = strtok(NULL, " ");
                    // >>> operator must be followed by a file name
                    if (file_name == NULL){
                        fprintf(stderr, "Parsing error after >>>!\n");
                        parsing_error = 1;
                        break;
                    }
                    append = 1;
                    reversed = 1;
                }
            }
            // Normal token
            if (file_name==NULL && background == 0){
                args[i] = token;
                i++;
            }
            token = strtok(NULL, " ");
        }  
        // Null terminate args
        args[i] = NULL;

        // If parsing error, don't execute command
        if (parsing_error){
            continue;
        }

        // Check for built-in commands
        if (strcmp(command, "exit") == 0){
            save_alias();
            exit(0);
        }
        else if (strcmp(command, "alias") == 0){
            // Alias must be followed by a name and =
            if (args[2] == NULL || strcmp(args[2], "=") != 0){
                fprintf(stderr, "Parsing error after alias!\n");
                continue;
            }
            // Set last executed
            last_executed[0] = '\0';
            strcpy(last_executed, last_candidate);

            create_alias(args);
            continue;
        }
        else if (strcmp(command, "bello")== 0){
            bello(file_name, append, reversed);
            continue;
        }

        // Check if it is an alias
        alias_no = find_from_alias(command, args, file_name, background, append, reversed);
        if (alias_no >= 0){
            continue;
        }
        // If not an alias, search in path
        find_from_path(command, args, file_name, background, append, reversed);
    }

    return 0;
}


; ahmet firat gamsiz
; 2020400180
; compiling: yes
; complete: yes
#lang racket


(provide (all-defined-out))

; read and parse the input file
(define parse (lambda (input-file)
        (letrec (
            [input-port (open-input-file input-file)]
            [read-and-combine (lambda ()
                (let ([line (read input-port)])
                    (if (eof-object? line)
                        '()
                        (append `(,line) (read-and-combine))
                    )
                )
            )]
            )
            (read-and-combine)
        )
    )
)
(define create-hash (lambda (vars values)
        (letrec (
            [create-hash-iter (lambda (vars values hash)
                (if (null? vars)
                    hash
                    (create-hash-iter (cdr vars) (cdr values) (hash-set hash (car vars) (car values)))
                )
            )]
            )
            (create-hash-iter vars values (hash))
        )
    )
)

(define add-to-hash (lambda (old-hash new-hash)
        (foldl (lambda (key hash) (hash-set hash key (hash-ref new-hash key)))
            old-hash
            (hash-keys new-hash)
        )
    )
)

(define eval-program (lambda (program-str)
        (get (eval-exprs (parse program-str) empty-state) '-r)
    )
)

; solution starts here
; 1. empty-state (5 points)
(define empty-state (hash))
; 2. get (5 points)
(define get (lambda (state var)
    (let ([val (hash-ref state var #f)])
        (if val
            val
            (eval var)))))
; 3. put (5 points)
(define put (lambda (state var val)
    (hash-set state var val)))
; 4. := (15 points)
(define := (lambda (var val-expr state)
    (let ([new-state (eval-expr val-expr state)])
        (put new-state var (get new-state '-r)))))
; 5. if: (15 points)
(define if: (lambda (test-expr then-exprs else-exprs state)
    (let ([new-state (eval-expr test-expr state)])
        (if (hash-ref new-state '-r #f)
            (eval-exprs then-exprs new-state)
            (eval-exprs else-exprs new-state)
        )
    )))
; 6. while: (15 points)
(define while: (lambda (test-expr body-exprs state)
    (let ([new-state (eval-expr test-expr state)])
        (if (hash-ref new-state '-r #f)
            (while: test-expr body-exprs (eval-exprs body-exprs new-state))
            new-state
        )
    )
    ))
; 7. func (15 points)
; needed workaround for foldl's implementation
(define alt-put (lambda (var val state)
    (hash-set state var val)))

; return the semi-lambda function with a state that has the parameters
(define func (lambda (params body-exprs state)
    (put state '-r (lambda vals
        (let ([new-state (foldl alt-put state params vals)])
        (get (eval-exprs body-exprs new-state) '-r)
    )))
))
; 8. eval-expr (20 points)
; recursively evaluate the list
(define map-eval (lambda (lst state) 
    (if (list? lst)
        (let ([oper (cons (first lst) '())] [params (rest lst)])
            (if (procedure? (get state (first lst)))
            (put state '-r (apply (get state (first lst)) (map (lambda (elem)
                (if (list? elem) 
                    (get (map-eval elem state) '-r)
                    (get state elem))
                ) params)
            ))
            (put state '-r (map (lambda (elem)
                eval (get state elem)
                ) lst)
            )
            )
        )
        (
            lst
        )
)))

; detect type of expression and evaluate accordingly
(define eval-expr (lambda (expr state)
    (cond 
        [(symbol? expr) (put state '-r (get state expr))]
        [(list? expr) 
            (cond 
            [(member (first expr) (list ':= 'if: 'while: 'func))
                (let ([state-lst (cons state '())]) 
                    (apply (eval (first expr)) (rest (append expr state-lst))))]
            [(eq? (first expr) 'lambda) (put state '-r (eval expr))]
            [else (map-eval expr state)]
            )]
        [else (put state '-r expr)]
)))
; 9 eval-exprs (5 points)
; evaluate the list of expressions and update the state
(define eval-exprs (lambda (exprs state) 
    (foldl eval-expr state exprs)
    ))

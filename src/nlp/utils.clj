(ns nlp.utils
  (:require
   [clojure.string :refer [lower-case capitalize]]
   [clojure.reflect :refer [type-reflect]]
   [medley.core :refer [find-first]]
   [camel-snake-kebab.core :refer [->kebab-case-keyword]]))

;;;
;;; Useful functions
;;;
(def atom? (complement coll?))

(defn find-in-coll [coll el]
  (find-first #(= % el) coll))

(defn make-keyword [s]
  (->kebab-case-keyword s))

(defmacro find-record [x]
  `(resolve '~x))

(defn find-record-field-set [record]
  (->> (:members (type-reflect record))
       (filter #(and (= (:type %) 'java.lang.Object)
                     (->> (:name %)
                          (name)
                          (re-find #"^__")
                          (not))
                     (= (:flags %) #{:public :final})))
       (mapv :name)
       (set)))

(defn multiple-nth [v indexes]
  (mapv #(nth v %) indexes))

;;;;;;;;;;;;;;;;;;;;;;
;; 0001 0010 0100 1000 [0] [1] [2] [3]
;; =>
;; 0011 0101 1001 [0 1] [0 2] [0 3]
;; 0110 1010      [1 2] [1 3]
;; 1100           [2 3]
;; =>
;; 0111 1011      [0 1 2] [0 1 3]
;; 1101           [0 2 3]
;; 1110           [1 2 3]
;; =>
;; 1111           [0 1 2 3]
;;
(defn powerset [col]
  (let [n (count col)]
    (letfn [(subset-position->next-positions
              ([]
               (mapv #(vector %) (range n)))
              ([pos-group]
               (mapv (fn [new-index]
                       (conj pos-group new-index))
                     (range (inc (last pos-group)) n))))]
      (loop [position-groups (subset-position->next-positions)
             result [nil]]
        (if (empty? position-groups)
          result
          (let [next-position-groups (mapcat (fn [group]
                                               (subset-position->next-positions group))
                                             position-groups)]
            (recur next-position-groups
                   (concat result (mapv #(multiple-nth col %) position-groups)))))))))

;;;
;;; strign -> sexp
;;;
(defn- ssexp->tokens [s idx]
  (loop [start idx end idx tokens []]
    (let [c (get s end)]
      (case c
        (\( \)) (let [paren-op (if (= c \() :lp :rp)]
                  (if (= start end)
                    [(conj tokens paren-op) (inc end)]
                    [(conj tokens (subs s start end) paren-op) (inc end)]))
        \space (if (= start end)
                 (recur (inc start) (inc start) tokens)
                 (recur (inc end) (inc end) (conj tokens (subs s start end))))
        (recur start (inc end) tokens)))))

#_
(defn- compute-next-stack [tokens initial-stack op-conv]
  (loop [[token & more-tokens] tokens
         stack initial-stack]
    (case token
      nil stack
      :rp (let [lp-index (.indexOf stack :lp)]
            (println "***"
                     lp-index
                     stack
                     (nthrest stack (inc lp-index))
                     (apply conj () (take lp-index stack)))
            (recur more-tokens (conj (nthrest stack (inc lp-index))
                                     (apply conj () (take lp-index stack)))))
      (recur more-tokens (conj stack token)))))

(defn- compute-next-stack [tokens initial-stack op-conv]
  (loop [[token & more-tokens] tokens
         stack initial-stack]
    (case token
      nil stack
      :rp (let [lp-index (.indexOf stack :lp)]
            (if (zero? lp-index)
              (recur more-tokens (rest stack))
              (recur more-tokens (conj (nthrest stack (inc lp-index))
                                       (apply conj () (take lp-index stack))))))
      (recur more-tokens (conj stack token)))))

(defn- compute-next-stack [tokens initial-stack op-conv]
  (loop [[token & more-tokens] tokens
         stack initial-stack]
    (case token
      nil stack
      :rp (let [lp-index (.indexOf stack :lp)]
            (if (zero? lp-index)
              (recur more-tokens (rest stack))
              (let [sexp (apply conj () (take lp-index stack))]
                (recur more-tokens (conj (nthrest stack (inc lp-index))
                                         (if (list? sexp)
                                           (cons (op-conv (first sexp))
                                                 (rest sexp))
                                           sexp))))))
      (recur more-tokens (conj stack token)))))

(defn str-sexp->sexp
  ([s]
   (str-sexp->sexp s identity))
  ([s op-conv]
   (let [max-index (count s)]
     (loop [index 0 stack ()]
       (if (<= max-index index)
         (if (empty? (rest stack))
           (first stack)
           (throw (Throwable. "Unmatched parens!")))
         (let [[tokens next-index] (ssexp->tokens s index) ]
           (recur next-index (compute-next-stack tokens stack op-conv))))))))


(ns nlp.utils
  (:require
   [clojure.string :refer [lower-case capitalize]]
   [clojure.reflect :refer [type-reflect]]
   [medley.core :refer [find-first]]
   [camel-snake-kebab.core :refer [->kebab-case-keyword]]))

;;;
;;; Useful functions
;;;
(defn find-in-coll [coll el]
  (find-first #(= % el) coll))

(defn make-keyword [s]
  (->kebab-case-keyword s))

(defmacro find-record [x]
  `(resolve '~x))

(defn fields->record-name-symbol [symv postfix]
  {:pre [(vector? symv)]}
  (->> (conj symv postfix)
       (mapv #(-> (name %) (capitalize)))
       (apply str)
       (symbol)))

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

;;;;;;;;;;;;;;;;;;;;;;
;;; [#{1} #{2} #{3} #{4}]
;;; =>   #{1}   [#{2} #{3} #{4}]
;;; ==>  #{2}   [#{3} #{4}]
;;; ===> #{3}   [#{4}]
;;; <=== #{3 4}
;;; <==  #{2 3} #{2 4}
;;; <=   #{1 2} #{1 3} #{1 4}
;;;
;;; [#{1 2} #{1 3} #{1 4} #{2 3} #{2 4} #{3 4}      #{1} #{2} #{3} #{4}]
;;;
;;; [#{1 2} #{1 3} #{1 4} #{2 3} #{2 4} #{3 4}]
;;; => #{1 2} [#{3} #{4}]
;;; <= #{1 2 3} #{1 2 4}
;;; => #{1 3} [#{4}]
;;; <= #{1 3 4}
;;;
;;;[#{1 2 3} #{1 2 4} #{1 3 4}]
;;;=> #{1 2 3} [#{4}]
;;;<= #{1 2 3 4}

(defn non-empty-subsets
  ([s]
   {:pre [(set? s)]}
   (let [result (mapv hash-set s)]
     (non-empty-subsets result (rest result) (rest result) [] result)))
  ([[seed-set & more-sets] target-sets init-targets collected-subsets result]
   (cond (and (empty? target-sets) (empty? collected-subsets)) result
         (empty? target-sets) (recur collected-subsets
                                     (rest init-targets)
                                     (rest init-targets)
                                     []
                                     (apply conj result collected-subsets))
         :else (let [subsets (mapv #(apply conj seed-set %) target-sets)]
                 (recur more-sets
                        (rest target-sets)
                        init-targets
                        (apply conj collected-subsets subsets)
                        result)))))



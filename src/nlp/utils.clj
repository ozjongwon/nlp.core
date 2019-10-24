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


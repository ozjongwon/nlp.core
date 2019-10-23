(ns nlp.records
  (:require
   [clojure.set :refer [union]]
   [nlp.utils :refer [fields->record-name-symbol find-record-field-set]])
  (:import
   [edu.stanford.nlp.ling CoreAnnotations$LemmaAnnotation

    ;; CoreAnnotations$SentencesAnnotation CoreAnnotations$TextAnnotation
    ;; CoreAnnotations$NamedEntityTagAnnotation CoreAnnotations$TokensAnnotation
    ;; CoreAnnotations$PartOfSpeechAnnotation
    ;; CoreAnnotations$MentionsAnnotation CoreAnnotations$NamedEntityTagAnnotation
    ;; CoreLabel TaggedWord Word SentenceUtils
    ]
   ))

;;; Protocols, records, etc
(defn token-ann->token-map [token-ann]
  {:token (.word token-ann)
   :begin (.beginPosition token-ann)
   :end (.endPosition token-ann)})

(defprotocol TokenBasedResult
  (make-token-result [this token-ann]))

(defrecord TokenizeResult [token begin end]
  TokenBasedResult
  (make-token-result [this token-ann]
    (merge this (token-ann->token-map token-ann))))

;;; Record extension

(defmacro def-tokenize-based-record [keys make-token-result-fn]
  (let [fields (->> (set keys) (sort) (mapv #(-> (name %) (symbol))))
        record-name (fields->record-name-symbol fields "Result")
        base-field-set (find-record-field-set TokenizeResult)]
    `(when-not (resolve '~record-name)
       (defrecord ~record-name ~(->> (union base-field-set fields)
                                     (vec))
         TokenBasedResult
         ~make-token-result-fn))))


(def-tokenize-based-record [:lemma]
  (make-token-result [this token-ann]
    (-> this
        (assoc :lemma (.get token-ann CoreAnnotations$LemmaAnnotation))
        (merge (token-ann->token-map token-ann)))))

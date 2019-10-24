(ns nlp.records
  (:require
   [clojure.set :refer [union]]
;;   [medley.core :refer [assoc-some]]
   [nlp.utils :refer [fields->record-name-symbol find-record-field-set]])
  (:import
   [edu.stanford.nlp.ling CoreAnnotations$LemmaAnnotation CoreAnnotations$TokensAnnotation

    ;; CoreAnnotations$SentencesAnnotation CoreAnnotations$TextAnnotation
    ;; CoreAnnotations$NamedEntityTagAnnotation
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

;;;
;;; Design Idea
;;;
;;; Stanford NLP pipeline has many annotations(see key->property-dependency in core.clj)
;;; and they have some properties:
;;; tokenize - token, begin, end
;;; ssplit - N/A, for sentence detection
;;; pos - TBD
;;; parse - TBD
;;; lemma - tokenize properties + lemma
;;; regexner - TBD
;;; depparse - TBD
;;; ner - tokenize properties or lemma properties + ner
;;; entitylink - TBD
;;; sentiment - TBD
;;; dcoref - TBD
;;; coref - TBD
;;; kbp - TBD
;;; quote - TBD
;;;

(defprotocol TokenBasedResult
  (%make-token-result [this token-ann])
  (token-based-result->annotation-class [this]))

(defonce result-prototypes (atom {}))

(defmacro get-result-prototype [result-type]
  `(or (get @result-prototypes ~result-type)
       (let [prototype# (eval `(. ~~result-type create {}))]
         (swap! result-prototypes assoc ~result-type prototype#)
         prototype#)))

(defn make-token-result [proto-class token-ann]
  (%make-token-result (get-result-prototype proto-class) token-ann))

(defrecord TokenizeResult [token begin end]
  TokenBasedResult
  (%make-token-result [this token-ann]
    (merge this (token-ann->token-map token-ann)))
  (token-based-result->annotation-class [this]
    CoreAnnotations$TokensAnnotation))

(defrecord LemmaResult [token begin end lemma]
  TokenBasedResult
  (%make-token-result [this token-ann]
    (let [token-result (make-token-result TokenizeResult token-ann)
          lemma (.get token-ann CoreAnnotations$LemmaAnnotation)]
      (if (= (:token token-result) lemma)
        token-result
        (merge this (assoc token-result :lemma lemma)))))
  (token-based-result->annotation-class [this]
    CoreAnnotations$TokensAnnotation))

(defrecord NerResult [token begin end lemma]
  TokenBasedResult
  (%make-token-result [this token-ann]
    (let [token-based-result (make-token-result LemmaResult token-ann)])
    ;; FIXME
    (merge this (token-ann->token-map token-ann)))
  (token-based-result->annotation-class [this]
    CoreAnnotations$TokensAnnotation))

;;; Record extension
#_
(defmacro def-tokenize-based-record [keys & protocol-impl-fns]
  (let [fields (->> (set keys) (sort) (mapv #(-> (name %) (symbol))))
        record-name (fields->record-name-symbol fields "Result")
        base-field-set (find-record-field-set TokenizeResult)]
    `(when-not (resolve '~record-name)
       (defrecord ~record-name ~(->> (union base-field-set fields)
                                     (vec))
         TokenBasedResult
         ~@protocol-impl-fns))))

#_
(def-tokenize-based-record [:lemma]
  (%make-token-result [this token-ann]
                     (let [token-map (token-ann->token-map token-ann)
                           lemma (.get token-ann CoreAnnotations$LemmaAnnotation)]
                       (if (= (:token token-map) lemma)
                         (map->TokenizeResult token-map)
                         (merge this (assoc token-map :lemma lemma)))))
  (token-based-result->annotation-class [this]
                                        CoreAnnotations$TokensAnnotation))
#_
(def-tokenize-based-record [:ner]
  (%make-token-result [this token-ann]
                     (-> this
                         (assoc :ner (.get token-ann CoreAnnotations$LemmaAnnotation))
                         (merge (token-ann->token-map token-ann))))
  (token-based-result->annotation-class [this]
                                        CoreAnnotations$TokensAnnotation))

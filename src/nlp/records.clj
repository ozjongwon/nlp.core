(ns nlp.records
  (:require
   [clojure.set :refer [union]]
;;   [medley.core :refer [assoc-some]]
   [nlp.utils :refer [fields->record-name-symbol find-record-field-set
                      make-keyword]])
  (:import
   [edu.stanford.nlp.ling CoreAnnotations$LemmaAnnotation CoreAnnotations$TokensAnnotation
    CoreAnnotations$PartOfSpeechAnnotation
    CoreAnnotations$MentionsAnnotation CoreAnnotations$TextAnnotation
    CoreAnnotations$TokenBeginAnnotation CoreAnnotations$TokenEndAnnotation
    CoreAnnotations$NamedEntityTagAnnotation

    ;; CoreAnnotations$SentencesAnnotation
    ;;
    ;; CoreAnnotations$MentionsAnnotation CoreAnnotations$NamedEntityTagAnnotation
    ;; CoreLabel TaggedWord Word SentenceUtils
    ]
   ))

;;; Protocols, records, etc
(defonce special-token-map {"``" :dqs ;; double quote start
                            "''" :dqe ;; double quote end
                            "`"  :sqs ;; single quote start
                            "'"  :sqe ;; single quote end
                            "-LRB-" :op ;; open paren
                            "-RRB-" :cp ;; close paren
                            "." :fs ;; full stop
                            "?" :q
                            "!" :e ;; exclamation mark
                            "..." :ep ;; Ellipsis points
                            ":" :c ;; colon
                            })
(defn- token-or-special-token [token]
  (or (get special-token-map token) token))

(defn- token-ann->token [token-ann]
  (token-or-special-token (.get token-ann CoreAnnotations$TextAnnotation)))

(defn token-ann->token-map [token-ann]
  {:token (token-ann->token token-ann) ;;(.word token-ann)
   :begin (.get token-ann CoreAnnotations$TokenBeginAnnotation) ;;(.beginPosition token-ann)
   :end (.get token-ann CoreAnnotations$TokenEndAnnotation) ;;(.endPosition token-ann)
   })

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
       ;; Uh... eval!
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

(defrecord PosResult [token begin end pos]
  TokenBasedResult
  (%make-token-result [this token-ann]
    (let [token-result (make-token-result TokenizeResult token-ann)
          pos (.get token-ann CoreAnnotations$PartOfSpeechAnnotation)]
      (merge this (assoc token-result :pos (or (get special-token-map pos) (make-keyword pos))))))
  (token-based-result->annotation-class [this]
    CoreAnnotations$TokensAnnotation))

(defrecord LemmaResult [token begin end lemma]
  TokenBasedResult
  (%make-token-result [this token-ann]
    (let [token-result (make-token-result PosResult token-ann)
          lemma (.get token-ann CoreAnnotations$LemmaAnnotation)]
      (if (or (nil? lemma) (= (:token token-result) lemma))
        token-result
        (merge this (assoc token-result :lemma lemma)))))
  (token-based-result->annotation-class [this]
    CoreAnnotations$TokensAnnotation))

;;; NerResult may have TokenNerResult which does not have lemma
;;; So the result can be: PosResult TokenNerResult, or NerResult
(defrecord PosNerResult [token begin end pos ner])

(defrecord LemmaNerResult [token begin end pos lemma ner])

(defn- get-ner-map-constructor [token-result]
  (condp = (class token-result)
    PosResult  map->PosNerResult
    LemmaResult map->LemmaNerResult))

(defn- make-ner-result [mention-ann]
  (let [token-result (make-token-result LemmaResult mention-ann)
        ner (.get mention-ann CoreAnnotations$NamedEntityTagAnnotation)]
    (if (or (= ner "O")  (nil? ner))
      token-result
      (-> (assoc token-result :ner (make-keyword ner))
          ((get-ner-map-constructor token-result))))))

(defrecord NerResult [token begin end lemma]
  TokenBasedResult
  (%make-token-result [this mention-ann]
    (make-ner-result mention-ann))
  (token-based-result->annotation-class [this]
    CoreAnnotations$TokensAnnotation))


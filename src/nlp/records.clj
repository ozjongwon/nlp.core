(ns nlp.records
  (:require
   [clojure.set :refer [union]]
   [clojure.string :refer [join]]
   [camel-snake-kebab.core :refer [->PascalCaseSymbol ->kebab-case-symbol]]
   [nlp.utils :refer [make-keyword powerset find-record-field-set]])
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

(defrecord AnnotationInfo [key dependency annotation-class result-converter])

(defn- remove-other-ner-value [ner-val]
  (when-not (= ner-val "O")
    ner-val))

(defonce annotation-info-vector
  (let [infov (mapv #(apply ->AnnotationInfo %)
                    [[:tokenize [] nil nil]
                     [:docdate [] nil nil]
                     [:cleanxml ["tokenize"] nil nil]
                     [:ssplit ["tokenize"] nil nil]
                     [:pos ["tokenize" "ssplit"] CoreAnnotations$PartOfSpeechAnnotation nil]
                     [:parse ["tokenize" "ssplit"] nil nil]
                     [:lemma ["tokenize" "ssplit" "pos"] CoreAnnotations$LemmaAnnotation nil]
                     [:regexner ["tokenize" "ssplit" "pos"] nil nil]
                     [:depparse ["tokenize" "ssplit" "pos"] nil nil]
                     [:ner ["tokenize" "ssplit" "pos" "lemma"]
                      CoreAnnotations$NamedEntityTagAnnotation remove-other-ner-value]
                     [:entitylink ["tokenize" "ssplit" "pos" "lemma"  "ner"] nil nil]
                     [:sentiment ["tokenize" "ssplit" "pos" "parse"] nil nil]
                     [:dcoref ["tokenize" "ssplit" "pos" "lemma"  "ner" "parse"] nil nil]
                     [:coref ["tokenize" "ssplit" "pos" "lemma"  "ner"] nil nil]
                     [:kbp ["tokenize" "ssplit" "pos" "lemma"] nil nil]
                     [:quote ["tokenize" "ssplit" "pos" "lemma" "ner" "depparse"] nil nil]])]
    (zipmap (mapv :key infov) infov)))

(defn key->property-dependency [k]
  (-> (get annotation-info-vector k)
      (:dependency)))
#_
(defonce key->property-dependency
  {:tokenize []
   :docdate []
   :cleanxml ["tokenize"]
   :ssplit ["tokenize"]                 ; sentence detection
   :pos ["tokenize" "ssplit"]
   :parse ["tokenize" "ssplit"]
   :lemma ["tokenize" "ssplit" "pos"]
   :regexner ["tokenize" "ssplit" "pos"]
   :depparse ["tokenize" "ssplit" "pos"]
   :ner ["tokenize" "ssplit" "pos" "lemma"] ; Named Entity Recognition
   :entitylink ["tokenize" "ssplit" "pos" "lemma"  "ner"]
   :sentiment ["tokenize" "ssplit" "pos" "parse"]
   :dcoref ["tokenize" "ssplit" "pos" "lemma"  "ner" "parse"]
   :coref ["tokenize" "ssplit" "pos" "lemma"  "ner"]
   :kbp ["tokenize" "ssplit" "pos" "lemma"]
   :quote ["tokenize" "ssplit" "pos" "lemma" "ner" "depparse"]})

(defn sort-msk-lsk [ks] ;; most specific key -> least specific key
  (sort #(> (count (key->property-dependency %1))
            (count (key->property-dependency %2)))
        ks))

;;; Protocols, records, etc
(defonce special-token-map {"``" :double-quote-start
                            "''" :double-quote-end
                            "`"  :single-quote-start
                            "'"  :single-quote-end
                            "-LRB-" :open-paren
                            "-RRB-" :close-paren
                            "." :full-stop
                            "?" :question-mark
                            "!" :exclamation-mark
                            "..." :ellipsis-points
                            ":" :colon
                            "," :comma
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
  (prototype->make-operation-result [this token-ann])
  (prototype->annotation-class [this]))

(defonce result-prototypes (atom {}))

(defmacro get-result-prototype [result-type]
  `(or (get @result-prototypes ~result-type)
       ;; Uh... eval!
       (let [prototype# (eval `(. ~~result-type create {}))]
         (swap! result-prototypes assoc ~result-type prototype#)
         prototype#)))

(defn make-operation-result [proto-class token-ann]
  (prototype->make-operation-result (get-result-prototype proto-class) token-ann))

;;; Tokenize records
(defrecord TokenizeResult [token begin end]
  TokenBasedResult
  (prototype->make-operation-result [this token-ann]
    (merge this (token-ann->token-map token-ann)))
  (prototype->annotation-class [this]
    CoreAnnotations$TokensAnnotation))

;;; Other records
(defn operation-keys->result-record-symbol [key-set]
  (->> (conj (mapv name (sort-msk-lsk key-set)) "result")
       (join \-)
       (camel-snake-kebab.core/->PascalCaseSymbol)))

(let [records-ns *ns*]
  (defn operation-keys->result-record [key-set]
    (->> (operation-keys->result-record-symbol key-set)
         (ns-resolve records-ns))))

(defn record-key->record-slots [k]
  (if-let [record (operation-keys->result-record [k])]
    (find-record-field-set record)
    [(->kebab-case-symbol k)]))

#_
(defonce key->core-annotation-class
  {:pos   CoreAnnotations$PartOfSpeechAnnotation
   :lemma CoreAnnotations$LemmaAnnotation
   :ner   CoreAnnotations$NamedEntityTagAnnotation})

(defmulti make-protocol-for (fn [protocol & _] protocol))


(defn prototype->exec-operation [val-annotation annotation-class converter]
  (let [result (.get val-annotation annotation-class)]
    (if converter
      (converter result)
      result)))

(defmethod make-protocol-for 'TokenBasedResult [protocol kset slots]
  (let [this (gensym)
        token-ann (gensym)
        [k & super-ks] (sort-msk-lsk kset)
        {:keys [annotation-class result-converter]} (get annotation-info-vector k)
        %make-body (if (nil? super-ks)
                     `(assoc ~this ~k (prototype->exec-operation ~token-ann
                                                                 ~annotation-class
                                                                 ~result-converter))
                     (let  [super-name (operation-keys->result-record-symbol super-ks)]
                       `(assoc (merge ~this (make-operation-result ~super-name ~token-ann))
                               ~k (prototype->exec-operation ~token-ann
                                                             ~annotation-class
                                                             ~result-converter))))]
    `(TokenBasedResult
      (prototype->make-operation-result [~this ~token-ann]
         ~%make-body)
      (prototype->annotation-class [_#]
         CoreAnnotations$TokensAnnotation))))

(defn maybe-define-result-record [protocol kset]
  (if (or (empty? kset) (operation-keys->result-record kset))
    ;; record exists
    nil
    ;; a new record
    (let [record-symbol (operation-keys->result-record-symbol kset)
          record-slots (sort (set (mapcat #(record-key->record-slots %) kset)))]
      `((defrecord ~record-symbol [~@record-slots]
          ~@(make-protocol-for protocol kset record-slots))))))

(defmacro nlp-result-records [protocol ks]
  `(do
     ~@(mapcat #(maybe-define-result-record protocol %) (powerset ks))))

;;;
;;; Define result records
;;;
(nlp-result-records TokenBasedResult [:ner :lemma :pos :tokenize])

;; (macroexpand '(nlp-result-records TokenBasedResult [:ner :lemma :pos :tokenize]))

;; (defn- get-ner-map-constructor [token-result]
;;   (condp = (class token-result)
;;     PosResult  map->PosNerResult
;;     LemmaResult map->LemmaNerResult))

;; (defn- make-ner-result [mention-ann]
;;   (let [token-result (make-operation-result LemmaResult mention-ann)
;;         ner (.get mention-ann CoreAnnotations$NamedEntityTagAnnotation)]
;;     (if (or (= ner "O")  (nil? ner))
;;       token-result
;;       (-> (assoc token-result :ner (make-keyword ner))
;;           ((get-ner-map-constructor token-result))))))

;; (defrecord NerResult [token begin end lemma]
;;   TokenBasedResult
;;   (prototype->make-operation-result [this subkeys mention-ann]
;;     (make-ner-result mention-ann))
;;   (prototype->annotation-class [this]
;;     CoreAnnotations$TokensAnnotation))


#_
(defrecord SentimentResult [score]
  SentenceBasedResult
  (prototype->make-operation-result [this subkeys token-ann]
    (let [token-result (make-operation-result PosResult token-ann)
          lemma (.get token-ann CoreAnnotations$LemmaAnnotation)]
      (if (or (nil? lemma) (= (:token token-result) lemma))
        token-result
        (merge this (assoc token-result :lemma lemma)))))
  (prototype->annotation-class [this]
    CoreAnnotations$TokensAnnotation))

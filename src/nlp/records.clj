(ns nlp.records
  (:require
   [clojure.set :refer [union]]
   [clojure.string :refer [join]]
   [camel-snake-kebab.core :refer [->PascalCaseSymbol ->kebab-case-symbol]]
   [nlp.utils :refer [make-keyword powerset find-record-field-set]])
  (:import
   [edu.stanford.nlp.ling CoreAnnotations$LemmaAnnotation
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

(defrecord AnnotationInfo [key operation-level dependency operation-class result-converter
                           ;; FIXME:  sentence, token, etc
                           ])

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

(defn- transform-ner-value [ner-val]
  (when-not (= ner-val "O")
    (make-keyword ner-val)))

(defn- transform-lemma-value [val]
  (or (get special-token-map val) val))

(defn- transform-pos-value [val]
  (or (get special-token-map val) (make-keyword val)))

(defonce annotation-info-map
  (let [infov (mapv #(apply ->AnnotationInfo %)
                    [[:tokenize :token [] nil nil]
                     [:docdate nil [] nil nil]
                     [:cleanxml nil ["tokenize"] nil nil]
                     [:ssplit nil ["tokenize"] nil nil]
                     [:pos :token ["tokenize" "ssplit"]
                      CoreAnnotations$PartOfSpeechAnnotation transform-pos-value]
                     [:parse nil ["tokenize" "ssplit"] nil nil]
                     [:lemma :token ["tokenize" "ssplit" "pos"]
                      CoreAnnotations$LemmaAnnotation transform-lemma-value]
                     [:regexner nil ["tokenize" "ssplit" "pos"] nil nil]
                     [:depparse nil ["tokenize" "ssplit" "pos"] nil nil]
                     [:ner :token ["tokenize" "ssplit" "pos" "lemma"]
                      CoreAnnotations$NamedEntityTagAnnotation transform-ner-value]
                     [:entitylink nil ["tokenize" "ssplit" "pos" "lemma"  "ner"] nil nil]
                     [:sentiment :sentence ["tokenize" "ssplit" "pos" "parse"] nil nil]
                     [:dcoref nil ["tokenize" "ssplit" "pos" "lemma"  "ner" "parse"] nil nil]
                     [:coref nil ["tokenize" "ssplit" "pos" "lemma"  "ner"] nil nil]
                     [:kbp nil ["tokenize" "ssplit" "pos" "lemma"] nil nil]
                     [:quote nil ["tokenize" "ssplit" "pos" "lemma" "ner" "depparse"] nil nil]])]
    (zipmap (mapv :key infov) infov)))

(defn annotators-keys->op-dispatch-set [annotators-keys]
  (reduce-kv #(assoc %1 %2 (set %3))
             {}
             (group-by #(:operation-level %)
                       (vals (select-keys annotation-info-map annotators-keys)))))

(defn key->property-dependency [k]
  (-> (get annotation-info-map k)
      (:dependency)))

(defn sort-msk-lsk [ks] ;; most specific key -> least specific key
  (sort #(> (count (key->property-dependency %1))
            (count (key->property-dependency %2)))
        ks))

;;; Protocols, records, etc

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

(defprotocol OperationResult
  (prototype->make-token-based-operation-result [this token-ann]))

(defonce result-prototypes (atom {}))

(defmacro result-class->prototype [result-type]
  `(or (get @result-prototypes ~result-type)
       ;; Uh... eval!
       (let [prototype# (eval `(. ~~result-type create {}))]
         (swap! result-prototypes assoc ~result-type prototype#)
         prototype#)))

(defn make-token-based-operation-result [proto-class token-ann]
  (prototype->make-token-based-operation-result (result-class->prototype proto-class) token-ann))

;;; Tokenize records
(defrecord TokenizeResult [token begin end]
  OperationResult
  (prototype->make-token-based-operation-result [this token-ann]
    (merge this (token-ann->token-map token-ann))))

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

;; (prototype->exec-token-based-operation #object[edu.stanford.nlp.ling.CoreLabel 0x59d032f0 "Joe-1"]
;;                            CoreAnnotations$PartOfSpeechAnnotation
;;                            #function[nlp.records/transform-pos-value])
(defn prototype->exec-token-based-operation [val-annotation operation-class converter]
  (let [result (.get val-annotation operation-class)]
    (if converter
      (converter result)
      result)))

(defn make-protocol-for-token-based [kset slots]
  (let [this (gensym)
        token-ann (gensym)
        [k & super-ks] (sort-msk-lsk kset)
        {:keys [operation-level operation-class result-converter]} (get annotation-info-map k)
        token-op-body (if (nil? super-ks)
                        `(assoc ~this ~k (prototype->exec-token-based-operation ~token-ann
                                                                                ~operation-class
                                                                                ~result-converter))
                        (let  [super-name (operation-keys->result-record-symbol super-ks)]
                          `(assoc (merge ~this (make-token-based-operation-result ~super-name ~token-ann))
                                  ~k (prototype->exec-token-based-operation ~token-ann
                                                                            ~operation-class
                                                                            ~result-converter))))]
    (when token-op-body
      `(OperationResult
        ~@(when token-op-body
            `((prototype->make-token-based-operation-result [~this ~token-ann]
                                                            ~token-op-body)))))))

(defn- info-set->base-records-parts [info-set]
  (when-let [kset-list (rest (powerset (mapv :key info-set)))]
    (mapcat (fn [kset]
            (if (operation-keys->result-record kset)
              ;; record exists
              nil
              ;; a new record
              [[(operation-keys->result-record-symbol kset)
                (sort (set (mapcat #(record-key->record-slots %) kset)))
                kset]]))
          kset-list)))

(defn define-sentence-based-records [sentence-info-set]
  (mapv (fn [[record-symbol record-slots]]
          (when (and record-symbol record-slots)
            `(defrecord ~record-symbol [~@record-slots])))
        (info-set->base-records-parts sentence-info-set)))

(defn define-token-based-records [token-info-set]
  (mapv (fn [[record-symbol record-slots kset]]
          (when-let [protocol-body (make-protocol-for-token-based kset record-slots)]
            `(defrecord ~record-symbol [~@record-slots]
               ~@protocol-body)))
        (info-set->base-records-parts token-info-set)))

(defmacro nlp-result-records [ks]
  ;; 1. define sentence based records
  ;; 2. define token based records
  (let [{:keys [token sentence]} (annotators-keys->op-dispatch-set ks)]
   `(do
      ~@(define-sentence-based-records sentence)
      ~@(define-token-based-records token))))

;;;
;;; Define result records
;;;
(nlp-result-records [:ner :lemma :pos :tokenize :sentiment])

;; (macroexpand '(nlp-result-records [:ner :lemma :pos :tokenize :sentiment]))

;; (defn- get-ner-map-constructor [token-result]
;;   (condp = (class token-result)
;;     PosResult  map->PosNerResult
;;     LemmaResult map->LemmaNerResult))

;; (defn- make-ner-result [mention-ann]
;;   (let [token-result (make-token-based-operation-result LemmaResult mention-ann)
;;         ner (.get mention-ann CoreAnnotations$NamedEntityTagAnnotation)]
;;     (if (or (= ner "O")  (nil? ner))
;;       token-result
;;       (-> (assoc token-result :ner (make-keyword ner))
;;           ((get-ner-map-constructor token-result))))))

;; (defrecord NerResult [token begin end lemma]
;;   OperationResult
;;   (prototype->make-token-based-operation-result [this subkeys mention-ann]
;;     (make-ner-result mention-ann)))


#_
(defrecord SentimentResult [score]
  SentenceBasedResult
  (prototype->make-token-based-operation-result [this subkeys token-ann]
    (let [token-result (make-token-based-operation-result PosResult token-ann)
          lemma (.get token-ann CoreAnnotations$LemmaAnnotation)]
      (if (or (nil? lemma) (= (:token token-result) lemma))
        token-result
        (merge this (assoc token-result :lemma lemma))))))

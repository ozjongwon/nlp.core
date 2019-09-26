(ns nlp.core
 (:require
  [clojure.string :refer [join]]
  [medley.core :refer [find-first]]
  ;;[clojure.reflect :refer [reflect]]
  )
  (:import
   [java.util Properties]
   [clojure.lang Reflector]
   ;;[clojure.reflect Constructor]
   ;;clojure.lang.Reflector
   [java.io StringReader]
   ;; Tokenizers
   ;;[edu.stanford.nlp.international.arabic.process ArabicTokenizer]
   ;;[edu.stanford.nlp.trees.international.pennchinese CHTBTokenizer]
   ;;[edu.stanford.nlp.international.french.process FrenchTokenizer]
   [edu.stanford.nlp.process WordTokenFactory CoreLabelTokenFactory
    DocumentPreprocessor PTBTokenizer TokenizerFactory
    ;; LexerTokenizer TokenizerAdapter WhitespaceTokenizer WordSegmentingTokenizer
    ]
   ;;[edu.stanford.nlp.ie.machinereading.domains.ace.reader RobustTokenizer]
   ;;[edu.stanford.nlp.international.spanish.process SpanishTokenizer]

   [edu.stanford.nlp.ling  CoreAnnotations$SentencesAnnotation
    CoreAnnotations$TextAnnotation
    CoreAnnotations$NamedEntityTagAnnotation
    CoreAnnotations$TokensAnnotation
    Word]
   [edu.stanford.nlp.pipeline Annotation StanfordCoreNLP]
   )
  (:gen-class :main true))

;;;
;;; Utility fns
;;;

(defn java-class->simple-name [cls]
  (.getSimpleName cls))
#_
(defn find-new-tokenizer [clss]
  (filter #(= (str "new" (java-class->simple-name clss))
              (name (:name %)))
          (:members (reflect clss))))

(defn make-print-method [obj]
  (let [resolved-obj (resolve obj)
        str-prefix (str "#<"  (.getSimpleName resolved-obj) " ")]
    `(defmethod print-method ~resolved-obj
       [piece# ^java.io.Writer writer#]
       (.write writer#
               (str ~str-prefix (.toString piece#) ">")))))

(defmacro def-print-methods [& objects]
  (let [qualified-objs objects]
    `(do ~@(map (fn [obj]
                  (make-print-method obj)) qualified-objs))))

(def-print-methods CoreLabel TaggedWord Word)

;;;
;;; Tokenizers
;;;
(defn- %make-ptb-tokenizer-args [k text options]
  (object-array [(StringReader. text)
                 (case k
                   :ptb-word (WordTokenFactory.)
                   :ptb-core-label (CoreLabelTokenFactory.))
                 options]))

(defonce key->new-tokenizer
  {:ptb-word {:class PTBTokenizer :args-fn (partial %make-ptb-tokenizer-args :ptb-word)}
   :ptb-core-label {:class PTBTokenizer :args-fn (partial %make-ptb-tokenizer-args :ptb-core-label)}})

(defn make-tokenizer
  ([k text]
   (make-tokenizer k text nil))
  ([k text options]
   (let [{:keys [class args-fn]} (key->new-tokenizer k)]
     (Reflector/invokeConstructor class (args-fn text options)))))

;; (defn tokenize
;;   ([k text]
;;    (tokenize k text nil))
;;   ([k text options]
;;    ;; FIXME: options is not being used, actually.
;;    (->> (mapv name options)
;;         (join \,)
;;         (make-tokenizer k text)
;;         (.tokenize)
;;         (mapv #(array-map :token (.value %)
;;                           :start-offset (.beginPosition %)
;;                           :end-offset (.endPosition %))))))

(defn tokenize [k text]
  (->> (make-tokenizer k text)
       (.tokenize)
       (mapv #(array-map :token (.value %)
                         :start-offset (.beginPosition %)
                         :end-offset (.endPosition %)))))

;; (tokenize :ptb-core-label "The meaning and purpose of life is plain to see.")
;; (tokenize :ptb-word "The meaning and purpose of life is plain to see.")

;;;
;;; sentences
;;;
(defn text->core-label-sentences [text]
  "Split a string into a sequence of sentences, each of which is a sequence of CoreLabels"
  (->> (StringReader. text)
       (DocumentPreprocessor.)
       (.iterator)
       (iterator-seq)))

;;;
;;; Pipeline
;;; https://stanfordnlp.github.io/CoreNLP/annotators.html
(defn find-in-coll [coll el]
  (find-first #(= % el) coll))

(defonce key->property-dependency
  {:tokenize []
   :docdate []
   :cleanxml ["tokenize"]
   :ssplit ["tokenize"]
   :pos ["tokenize" "ssplit"]
   :parse ["tokenize" "ssplit"]
   :lemma ["tokenize" "ssplit" "pos"]
   :regexner ["tokenize" "ssplit" "pos"]
   :depparse ["tokenize" "ssplit" "pos"]
   :ner ["tokenize" "ssplit" "pos" "lemma"]
   :entitylink ["tokenize" "ssplit" "pos" "lemma"  "ner"]
   :sentiment ["tokenize" "ssplit" "pos" "parse"]
   :dcoref ["tokenize" "ssplit" "pos" "lemma"  "ner" "parse"]
   :coref ["tokenize" "ssplit" "pos" "lemma"  "ner"]
   :kbp ["tokenize" "ssplit" "pos" "lemma"]
   :quote ["tokenize" "ssplit" "pos" "lemma" "ner" "depparse"]
   }
  )

(defn- %check-parse-dependeny-when-opt [args opt]
  (when (find-in-coll args opt)
    (let [parse? (find-in-coll args :parse)
          dparse? (find-in-coll args :dparse)]
      (when (and parse? dparse?)
        (throw (Exception. "Only one of parse or depparse is allowed")))
      (when (not (or parse? dparse?))
        (throw (Exception. "One of parse or depparse is required"))))))

(defn- maybe-check-coref-dependeny [args]
  (when (find-in-coll args :coref)
    (%check-parse-dependeny-when-opt args :coref)))

(defn- maybe-check-kbp-dependency [args]
  (when (find-in-coll args :kbp)
    (%check-parse-dependeny-when-opt args :kbp)
    (when-not (find-in-coll args :coref)
      (throw (Exception. "Option :coref is required")))))

(defn make-annotators-opts
  ([args]
   ;; Special case checking
   (maybe-check-coref-dependeny args)
   (maybe-check-kbp-dependency args)
   (make-annotators-opts args []))
  ([[k & more-ks] result]
   (if (nil? k)
     (distinct result)
     (if-let [opts-found (key->property-dependency k)]
       (recur more-ks (into result (conj opts-found (name k))))
       (throw (Exception. (str "Unknown key: " k "!")))))))

(defn make-pipeline [& annotators-args]
  (StanfordCoreNLP. (doto (Properties.)
                      (.put "annotators" (join \, (make-annotators-opts annotators-args))))
                    true))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; (defn make-tokenizer [k & args]
;;   (Reflector/invokeConstructor (key->tokenizer k) (into-array Object args)))


;; (defn find-constructor [clss]
;;   (find-first (fn [m]
;;                 (= (class m) Constructor))
;;               (:members (reflect clss))))

;; (defn find-all-constructors [clss]
;;   (filter (fn [m]
;;             (= (class m) Constructor))
;;           (:members (reflect clss))))

;;;
;;;
;;; (def pipeline (make-pipeline :parse))
;;; (def annotation (.process pipeline "The meaning and purpose of life is plain to see." ))
;;; (.annotate pipeline annotation)
;;; (.prettyPrint pipeline annotation *out*)
;;; https://nlp.stanford.edu/software/stanford-dependencies.shtml


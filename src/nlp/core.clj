(ns nlp.core
 ;; (:require
 ;;   [medley.core :refer [find-first]]
 ;;   [clojure.reflect :refer [reflect]])
  (:import
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

   [edu.stanford.nlp.ling CoreLabel TaggedWord Word]
   )
  (:gen-class :main true))

;;;
;;; Utility fns
;;;

(defn java-class->simple-name [cls]
  (.getSimpleName cls))

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

;; "americanize,normalizeAmpersandEntity=false"
;;     normalizeOtherBrackets: Whether to map other common bracket characters to -LCB-, -LRB-, -RCB-, -RRB-, roughly as in the Penn Treebank. Default is true.
;;     asciiQuotes: Whether to map all quote characters to the traditional ' and ". Default is false.
;;     latexQuotes: Whether to map quotes to ``, `, ', '', as in Latex and the PTB3 WSJ (though this is now heavily frowned on in Unicode). If true, this takes precedence over the setting of unicodeQuotes; if both are false, no mapping is done. Default is true.
;;     unicodeQuotes: Whether to map quotes to the range U+2018 to U+201D, the preferred unicode encoding of single and double quotes. Default is false.
;;     ptb3Ellipsis: Whether to map ellipses to three dots (...), the old PTB3 WSJ coding of an ellipsis. If true, this takes precedence over the setting of unicodeEllipsis; if both are false, no mapping is done. Default is true.
;;     unicodeEllipsis: Whether to map dot and optional space sequences to U+2026, the Unicode ellipsis character. Default is false.
;;     ptb3Dashes: Whether to turn various dash characters into "--", the dominant encoding of dashes in the PTB3 WSJ. Default is true.
;;     keepAssimilations: true to tokenize "gonna", false to tokenize "gon na". Default is true.
;;     escapeForwardSlashAsterisk: Whether to put a backslash escape in front of / and * as the old PTB3 WSJ does for some reason (something to do with Lisp readers??). Default is true.
;;     untokenizable: What to do with untokenizable characters (ones not known to the tokenizer). Six options combining whether to log a warning for none, the first, or all, and whether to delete them or to include them as single character tokens in the output: noneDelete, firstDelete, allDelete, noneKeep, firstKeep, allKeep. The default is "firstDelete".
;;     strictTreebank3: PTBTokenizer deliberately deviates from strict PTB3 WSJ tokenization in two cases. Setting this improves compatibility for those cases. They are: (i) When an acronym is followed by a sentence end, such as "U.K." at the end of a sentence, the PTB3 has tokens of "Corp" and ".", while by default PTBTokenizer duplicates the period returning tokens of "Corp." and ".", and (ii) PTBTokenizer will return numbers with a whole number and a fractional part like "5 7/8" as a single token, with a non-breaking space in the middle, while the PTB3 separates them into two tokens "5" and "7/8". (Exception: for only "U.S." the treebank does have the two tokens "U.S." and "." like our default; strictTreebank3 now does that too.) The default is false.
;;     splitHyphenated: whether or not to tokenize segments of hyphenated words separately ("school" "-" "aged", "frog" "-" "lipped"), keeping the exceptions in Supplementary Guidelines for ETTB 2.0 by Justin Mott, Colin Warner, Ann Bies, Ann Taylor and CLEAR guidelines (Bracketing Biomedical Text) by Colin Warner et al. (2012). Default is false, which maintains old treebank tokenizer behavior.


(defn make-tokenizer
  ([k text]
   (make-tokenizer k text nil))
  ([k text options]
   (let [{:keys [class args-fn]} (key->new-tokenizer k)]
     (Reflector/invokeConstructor class (args-fn text options)))))

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

(defn- check-coref-dependeny [args]
  (%check-parse-dependeny-when-opt args :coref))

(defn- check-kbp-dependency [args]
  (%check-parse-dependeny-when-opt args :kbp)
  (when-not (find-in-coll args :coref)
    (throw (Exception. "Option :coref is required"))))

(defn make-pipeline-properties
  ([args]
   ;; Special case checking
   (check-coref-dependeny args)
   (check-kbp-dependency args)
   (make-pipeline-properties args []))
  ([[k & more-ks] result]
   (if (nil? k)
     (distinct result)
     (if-let [opts-found (key->property-dependency k)]
       (recur more-ks (into result (conj opts-found (name k))))
       (throw (Exception. (str "Unknown key: " k "!")))))))

(defn make-pipeline [& args]
  (StanfordCoreNLP. (make-pipeline-properties args) true))


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

(defn make-text-tokenizer [k text]
  ((key->tokenizer k) text))



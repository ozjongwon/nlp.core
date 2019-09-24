(ns nlp.core
  #_
  (:require ;; [medley.core :refer [find-first]]
            ;;[clojure.reflect :refer [reflect]]
            ;;[clojure.lang.Reflector :refer [invokeConstructor]]
            )
  (:import
   ;;[clojure.reflect Constructor]
   ;;clojure.lang.Reflector
   [java.io StringReader]
   ;; Tokenizers
   ;;[edu.stanford.nlp.international.arabic.process ArabicTokenizer]
   ;;[edu.stanford.nlp.trees.international.pennchinese CHTBTokenizer]
   ;;[edu.stanford.nlp.international.french.process FrenchTokenizer]
   [edu.stanford.nlp.process PTBTokenizer TokenizerFactory
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
(defonce key->tokenizer
  { ;; :arabic ArabicTokenizer
   ;; :chtb CHTBTokenizer
   ;; :french FrenchTokenizer
   ;; :lexer LexerTokenizer
   :ptb (fn [text]
          (. (PTBTokenizer/factory false false)
             TokenizerFactory/getTokenizer
             (StringReader. text)))
   ;; :adapter TokenizerAdapter
   ;; :whitespace WhitespaceTokenizer
   ;; :word-segmenting WordSegmentingTokenizer
   ;; :robust RobustTokenizer
   ;; :spanish SpanishTokenizer
   })

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

;;;
;;; * NLP SUMMARY
;;;
;;; Application Areas of NLP
;;; - Searching
;;; - Machine Translation
;;; - Summation
;;; - Named Entity Recognition
;;; - Information Grouping
;;; - Part Of Speech tagging (POS)
;;; - Sentiment Analysis
;;; - Answering Queries
;;; - Speech recognition
;;; - Natural Language Generation
;;;
;;; Morpheme - the minimal unit of text which has meaning. Ex) prefixes and suffixes
;;; POS label - assign labels to words and morphemes.
;;; Stemming - finding the word stem.
;;; Lemmatization - determine the lemma, the base form
;;; Corereferences resolution - determine the relationship between words in sentences.
;;; Word Sense Disambiguation - find the intended meaning.
;;; Morphology - study of the structure of words
;;; Parts of Text - words, sentences, paragraphs, etc OR sometimes tokens == words
;;; Claassification - Assign labels to information found in text or documents:
;;;  - labels are known - classification.
;;;  - labels are unknown - clustering.
;;; Categorization - assign a text element to categories.
;;;
;;;-------------------------------------------------------------------------------------------
;;;
;;; * PTBTokenizer - CoreLabelTokenFactory Vs. WordTokenFactory
;;; * Annotators - extract relationship using Annotators with their operations
;;;  (aka text analytics). Ex) annotators == :tokenize :ssplit :parse ==> tokenize,
;;;  splits to sentences, and syntactic analysis by parsing
;;;
(ns nlp.core
  (:require
   [clojure.string :refer [join split]]
   [clojure.set :refer [intersection]]
   [medley.core :refer [find-first]])
  (:import
   [java.util Properties]
   [clojure.lang Reflector]
   ;;[clojure.reflect Constructor]
   ;;clojure.lang.Reflector
   [java.io StringReader]
   ;; Tokenizers
   ;;[edu.stanford.nlp.international.arabic.process ArabicTokenizer]
   ;;[edu.stanford.nlp.trees.international.pennchinese CHTBTokenizer]
   [edu.stanford.nlp.trees LabeledScoredTreeNode TreeCoreAnnotations$TreeAnnotation]
   [edu.stanford.nlp.semgraph SemanticGraphCoreAnnotations$EnhancedPlusPlusDependenciesAnnotation]
   ;;[edu.stanford.nlp.international.french.process FrenchTokenizer]
   [edu.stanford.nlp.process WordTokenFactory CoreLabelTokenFactory
    DocumentPreprocessor PTBTokenizer TokenizerFactory
    ;; LexerTokenizer TokenizerAdapter WhitespaceTokenizer WordSegmentingTokenizer
    ]
   ;;[edu.stanford.nlp.ie.machinereading.domains.ace.reader RobustTokenizer]
   ;;[edu.stanford.nlp.international.spanish.process SpanishTokenizer]

   [edu.stanford.nlp.ling CoreAnnotations$SentencesAnnotation CoreAnnotations$TextAnnotation
    CoreAnnotations$NamedEntityTagAnnotation CoreAnnotations$TokensAnnotation
    CoreAnnotations$PartOfSpeechAnnotation
    CoreLabel TaggedWord Word SentenceUtils]
   [edu.stanford.nlp.coref CorefCoreAnnotations$CorefChainAnnotation]
   ;; [edu.stanford.nlp.dcoref CorefCoreAnnotations$CorefChainAnnotation]
   [edu.stanford.nlp.pipeline Annotation StanfordCoreNLP]

   [edu.stanford.nlp.tagger.maxent MaxentTagger]
   [edu.stanford.nlp.parser.lexparser LexicalizedParser])
  (:gen-class :main true))


;;;
;;; StanfordCoreNLP Pipeline
;;; https://stanfordnlp.github.io/CoreNLP/annotators.html
;;;
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
   :quote ["tokenize" "ssplit" "pos" "lemma" "ner" "depparse"]})

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

(let [existing-core-nlp (atom nil)
      existing-opts-set (atom #{})]
  (defn make-pipeline [& annotators-keys]
    (let [annotators-opts (make-annotators-opts annotators-keys)
          annotators-opts-set (set annotators-opts)]
     (if (and @existing-core-nlp
              (= (intersection @existing-opts-set annotators-opts-set)
                 annotators-opts-set))
       @existing-core-nlp
       (let [new-core-nlp (StanfordCoreNLP. (doto (Properties.)
                                              (.put "annotators" (join \, annotators-opts)))
                                            true)]
         (reset! existing-core-nlp new-core-nlp)
         (reset! existing-opts-set annotators-opts-set)
         new-core-nlp)))))

;;;
;;; Annotator-key -> Operation
;;;
(defonce annotator-key->annotation-class
  {;; :tokenize
   ;; :docdate
   ;; :cleanxml
   :ssplit CoreAnnotations$SentencesAnnotation
   ;;:pos
   ;; :lemma
   ;; :kbp
   ;; :regexner
   ;; :ner
   ;; :entitylink
   ;; :coref
   :parse  TreeCoreAnnotations$TreeAnnotation
   ;; :dcoref
   ;; :sentiment
   ;; :depparse
   ;; :quote
   }
  )

(defrecord ParseResult [tree pos dependency])
(defprotocol AnnotationOperationResult
  (execute-operation [annotation-like]))

(def atom? (complement coll?))
(defn convert-tree
  ([tr kfn]
   (convert-tree tr kfn identity))
  ([[k v & more-nodes] kfn vfn]
   (if (nil? k)
     nil
     (cons (kfn k)
           (cons (if (atom? v)
                   (vfn v)
                   (convert-tree v kfn vfn))
                 (map #(convert-tree % kfn vfn) more-nodes))))))

(defn- tree->parse-tree [tree-node]
  (convert-tree (read-string (.toString tree-node)) keyword name))

(defn- tree->pos [tree-node]
  (->> (.taggedLabeledYield tree-node)
       (mapv #(vector (.word %) (keyword (.tag %))))
       (into (hash-map))))

(defn- tree->dependencies [tree-node]
  ;; FIXME: I don't know how to do this
  ;; s1 is a sentence not tree-node
  ;;(.get s1 SemanticGraphCoreAnnotations$EnhancedPlusPlusDependenciesAnnotation)
  )

(extend-protocol AnnotationOperationResult
  LabeledScoredTreeNode
  (execute-operation [this]
;;    (.labels this)
    ;; Dependency Parse (enhanced plus plus dependencies):
    ;; root(ROOT-0, Who-1)
    ;; cop(Who-1, are-2)
    ;; nsubj(Who-1, you-3)
    ;; punct(Who-1, ?-4)

    (->ParseResult (tree->parse-tree this)
                   (tree->pos this)
                   nil
                   #_
                   (tree->dependencies this))))

(defmulti execute-operation-using-sub-annotator (fn [_ sub-annotator] sub-annotator))
(defmethod execute-operation-using-sub-annotator TreeCoreAnnotations$TreeAnnotation [annotator sub-annotator]
  (mapv #(assoc (execute-operation (.get % sub-annotator))
                :dependency (.get % SemanticGraphCoreAnnotations$EnhancedPlusPlusDependenciesAnnotation))
        ;; FIXME:
        ;; https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/semgraph/SemanticGraph.html
        (.get annotator CoreAnnotations$SentencesAnnotation)))

(defrecord PerOperationResult [operation sentence-results])

;;;
;;; Main function
;;;
(defn analyse-text [text & annotators-keys]
  (let [annotation (Annotation. text)
        pipeline (apply make-pipeline annotators-keys)]
    (.annotate pipeline annotation) ;; side effect
    #_    [pipeline annotation]

    (mapv #(->PerOperationResult %
                                 (execute-operation-using-sub-annotator annotation
                                                                        (annotator-key->annotation-class %)))
          annotators-keys)))

;; (.prettyPrint pipeline annotation *out*)

;; Sentence #1 (4 tokens):
;; Who are you?

;; Tokens:
;; [Text=Who CharacterOffsetBegin=0 CharacterOffsetEnd=3 PartOfSpeech=WP]
;; [Text=are CharacterOffsetBegin=4 CharacterOffsetEnd=7 PartOfSpeech=VBP]
;; [Text=you CharacterOffsetBegin=8 CharacterOffsetEnd=11 PartOfSpeech=PRP]
;; [Text=? CharacterOffsetBegin=11 CharacterOffsetEnd=12 PartOfSpeech=.]

;; Constituency parse:
;; (ROOT
;;  (SBARQ
;;   (WHNP (WP Who))
;;   (SQ (VBP are)
;;       (NP (PRP you)))
;;   (. ?)))


;; Dependency Parse (enhanced plus plus dependencies):
;; root(ROOT-0, Who-1)
;; cop(Who-1, are-2)
;; nsubj(Who-1, you-3)
;; punct(Who-1, ?-4)
;; nil


;;;
;;; JVM option: -Xmx2g
;;;
(ns nlp.core
  (:require
   [clojure.string :refer [join split]]
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

   [edu.stanford.nlp.ling CoreAnnotations$SentencesAnnotation CoreAnnotations$TextAnnotation
    CoreAnnotations$NamedEntityTagAnnotation CoreAnnotations$TokensAnnotation
    CoreAnnotations$PartOfSpeechAnnotation
    CoreLabel TaggedWord Word SentenceUtils]
   ;; [edu.stanford.nlp.coref CorefCoreAnnotations$CorefChainAnnotation]
   [edu.stanford.nlp.dcoref CorefCoreAnnotations$CorefChainAnnotation]
   [edu.stanford.nlp.pipeline Annotation StanfordCoreNLP]

   [edu.stanford.nlp.tagger.maxent MaxentTagger]
   [edu.stanford.nlp.parser.lexparser LexicalizedParser]
   [edu.stanford.nlp.trees TreePrint]
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

(defn make-sentence-tokens [tokens]
  (mapv #(array-map :token (.word %)
                    :start-offset (.beginPosition %)
                    :end-offset (.endPosition %))
        tokens))

(defn tokenize [k text]
  (->> (make-tokenizer k text)
       (.tokenize)
       (make-sentence-tokens)))

;; (tokenize :ptb-core-label "The meaning and purpose of life is plain to see.")
;; (tokenize :ptb-word "The meaning and purpose of life is plain to see.")

;;;
;;; DocumentPreprocessor sentences
;;;
(defn text->core-label-sentences [text]
  "Split a string into a sequence of sentences, each of which is a sequence of CoreLabels"
  (->> (StringReader. text)
       (DocumentPreprocessor.)
       (.iterator)
       (iterator-seq)
       (mapv (fn [sentence-tokens]
               (make-sentence-tokens sentence-tokens)))))
;; (text->core-label-sentences "Let's pause, and then reflect.")

;;;
;;; StanfordCoreNLP Pipeline
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

#_
(defrecord PipelineAnnotatorKeys [pipeline annotator-keys])

#_
(defn make-pipeline-annotator-keys [& annotators-args]
  (let [annotators-opts (make-annotators-opts annotators-args)]
    (->PipelineAnnotatorKeys (StanfordCoreNLP. (doto (Properties.)
                                                 (.put "annotators" (join \, annotators-opts)))
                                               true)
                             (mapv keyword annotators-opts))))
#_
(defn make-pipeline [& annotators-args]
  (-> (apply make-pipeline-annotator-keys annotators-args)
      (:pipeline)))
(let [stanford-core-nlp (atom nil)]
 (defn make-pipeline [& annotators-keys]
   (StanfordCoreNLP. (doto (Properties.)
                       (.put "annotators" (join \, (make-annotators-opts annotators-keys))))
                     true)))

;;; (def pipeline (make-pipeline :ssplit))
;;; (def annotation (.process pipeline "The meaning and purpose of life is plain to see." ))
;;; (.annotate pipeline annotation)
;;; (.prettyPrint pipeline annotation *out*)
;;; https://nlp.stanford.edu/software/stanford-dependencies.shtml

;;; FIXME: don't know what to do yet
;;;
;;; (def lm (annotate-text "Eat, drink, and be merry, for life is but a dream" [:lemma] ))
;;; (def cll (.get lm CoreAnnotations$TokensAnnotation))
;;; (mapv #(.lemma %) cll)
;;;
;;;
;;; NOTE:
;;; - deal with sentences
;;; - Named Entity Recognition
;;;
(defprotocol AnnotationProtocol
  (token-entities [this annotation]))

(defrecord TextAnnotation [token start end]
  AnnotationProtocol
  (token-entities [_ annotation]
    (assoc :token (.get annotation CoreAnnotations$TextAnnotation)
           :start (.beginPosition annotation)
           :end (.endPosition annotation))))

(defrecord SentencesAnnotation []
  AnnotationProtocol
  )

(defonce annotation-keys->token-class
  {:text CoreAnnotations$TextAnnotation
   :sentences CoreAnnotations$SentencesAnnotation})

(defn annotate-text [text annotators-keys]
  ;;[text annotation-keys annotators-keys]
  (let [annotation (Annotation. text)]
    (.annotate (apply make-pipeline annotators-keys) annotation) ;; side effect
    annotation
    #_
    (mapv (fn [k]
            (let [clss (annotation-keys->token-class k)]
              (assert clss)
              {k (.get annotation clss)
               ;;:token (.word annotation)
               ;;:start-offset (.beginPosition annotation)
               ;;:end-offset (.endPosition annotation)
               }))
          annotation-keys)))

(defn- get-tokens-entities
  "builds map: {:token token :named-entity named-entity}"
  [tok-ann]
  {:token (.get tok-ann CoreAnnotations$TextAnnotation)
   :named-entity (.get tok-ann CoreAnnotations$NamedEntityTagAnnotation)
   :start-offset (.beginPosition tok-ann)
   :end-offset (.endPosition tok-ann)})

(defn- get-token-annotations
  "Passes TokenAnnotations extracted from SentencesAnnotation to get-tokens-entities
  which returns a map {:token token :named-entity ne}"
  [sentence-annotation]
  (mapv get-tokens-entities (.get sentence-annotation CoreAnnotations$TokensAnnotation)))

(defn- get-text-tokens [sen-ann]
  "builds map: {:tokens tokens}"
  {:tokens (get-token-annotations sen-ann)})

(defn- get-sentences-annotation
  "passes SentencesAnnotation extracted from Annotation object to function
  get-text-tokens which returns a map:
  {:tokens {:token token :named-entity ne}}"
  [^Annotation annotation]
  (mapv get-text-tokens (.get annotation CoreAnnotations$SentencesAnnotation)))

#_
(let [paragraph "Similar to stemming is Lemmatization. This is the process of finding its lemma, its form as found in a dictionary."
      pipeline (make-pipeline :lemma)]
  )

;; (def ann1 (abc "Similar to stemming is Lemmatization. This is the process of finding its lemma, its form as found in a dictionary." :lemma))
;; (.get ann1 CoreAnnotations$SentencesAnnotation)
;; (get-sentences-annotation ann1)


;;; POS Tagger
;;; (def tagger (MaxentTagger. "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger"))
;;; (def sl (. MaxentTagger tokenizeText (StringReader. "Similar to stemming is Lemmatization. This is the process of finding its lemma, its form as found in a dictionary.")))
;;; (.tagSentence tagger  (first sl))
;;; (mapv #(array-map :token (.word %) :tag (.tag %))  *1)


;;; POS tagging using pipeline
;;; (def ann (annotate-text "Similar to stemming is Lemmatization. This is the process of finding its lemma, its form as found in a dictionary." [:ner] ))
;;; (def sl (.get ann CoreAnnotations$SentencesAnnotation))
;;; (def cll (mapv #(.get % CoreAnnotations$TokensAnnotation) sl))
;; (mapv #(mapv (fn [cl]
;;                (array-map :word (.word cl)
;;                           :tag (.tag cl)
;;                           :ner (.ner cl)
;;                           :lemma (.lemma cl)))
;;              %)
;;       cll)


;;; Parser
;;; (LexicalizedParser/loadModel)
;;; ;;OR
;;; (def parser (LexicalizedParser/loadModel "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz" []))
;;; (def wl (SentenceUtils/toCoreLabelList (into-array ["The" "cow" "jumped" "over" "the" "moon" "." ])))
;;; (.parse parser wl)
;;; (read-string (.toString x))
;;; ;; OR
;;; (read-string (.pennString x))
;;;
;;;

(defn text->core-label-list [text]
  (->> (StringReader. text)
       (DocumentPreprocessor.)
       (.iterator)
       (iterator-seq)))
;;; Parse Sentences
;;; (def parser (LexicalizedParser/loadModel "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz" []))
;;; (def cll (text->core-label-list "Bell, based in Los Angeles, makes and distributes electronic, computer and building prod
;;; -ucts")
;;; (def trees (mapv #(.parse parser %) cll))
;;; (mapv #(read-string (.pennString %)) trees)
;;;
;;; formatString - A comma separated list of ways to print each Tree. For instance, "penn" or "words,typedDependencies". Known formats are: oneline, penn, latexTree, xmlTree, words, wordsAndTags, rootSymbolOnly, dependencies, typedDependencies, typedDependenciesCollapsed, collocations, semanticGraph, conllStyleDependencies, conll2007. The last two are both tab-separated values formats. The latter has a lot more columns filled with underscores. All of them print a blank line after the output except for oneline. oneline is also not meaningful in XML output (it is ignored: use penn instead). (Use of typedDependenciesCollapsed is deprecated. It works but we recommend instead selecting a type of dependencies using the optionsString argument. Note in particular that typedDependenciesCollapsed does not do CC propagation, which we generally recommend.)
;;; (def tp (TreePrint. "typedDependenciesCollapsed"))
;;; (def tp2 (TreePrint. "penn,typedDependenciesCollapsed"))
;;; (mapv #(.printTree tp %) trees)
;;; (mapv #(.printTree tp2 %) trees)
;;; (mapv #(read-string (.pennString %)) trees)

;;;
;;; Word dependencies
;;; (def sentence  "The cow jumped over the moon.")
;;; (def parse-tree (.parse parser (.tokenize (make-tokenizer :ptb-core-label sentence))))
;;; (def tlp (.treebankLanguagePack parser))
;;; (def gsf (.grammaticalStructureFactory tlp))
;;; (def gs (.newGrammaticalStructure gsf parse-tree))
;;; (def tdl (.typedDependenciesCCprocessed gs))
;;; (mapv #(array-map :governor-word (.toString (.gov %))
;;;                   :relation (-> (.reln %) (.getLongName))
;;;                   :dependent-word (.toString (.dep %)))
;;;       tdl)

;;;
;;; Coreference
;;; (def ann (annotate-text "He took his cash and she took her change and together they bought their lunch." [:parse :coref]))
;;; (def chains (into [] (.values (.get ann CorefCoreAnnotations$CorefChainAnnotation))))
;;; (def chains (->> (.get ann CorefCoreAnnotations$CorefChainAnnotation)
;;;                  (.values)
;;;                  (mapv #(.toString %))))
;;;(def ann2 (annotate-text "He took his cash and she took her change and together they bought their lunch." [:dcoref]))
;;; (def chains2 (->> (.get ann2 edu.stanford.nlp.coref.CorefCoreAnnotations$CorefChainAnnotation)
;;;                   (.values)
;;;                   (mapv #(.toString %))))
;;;


;; (def clusters (->> (.get ann2 CorefCoreAnnotations$CorefChainAnnotation)
;;                    (.values)
;;                    (mapv make-cluster)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defrecord Mention [context type start end])
(defn make-mention [mention]
  (->Mention (.toString mention)
             (.name (.mentionType mention))
             (.startIndex mention)
             (.endIndex mention)))

(defrecord Cluster [id mention-span gender mentions])
(defn make-cluster [coref-chain]
  (let [mention (.getRepresentativeMention coref-chain)
        mention-span (.mentionSpan mention)
        mentions (.getMentionsInTextualOrder coref-chain)]
    (->Cluster (.getChainID coref-chain)
               mention-span
               (.name (.gender mention))
               (mapv make-mention (.getMentionsInTextualOrder coref-chain)))))

(defn text->coref [text]
  (let [ann (annotate-text text [:parse :coref])]
    (->>  (.get ann edu.stanford.nlp.coref.CorefCoreAnnotations$CorefChainAnnotation)
          (.values)
          (mapv make-cluster))))

(defn text->dcoref [text]
  ;; FIXME: this does not work!
  (let [ann (annotate-text text [:parse :dcoref])]
    (->>  (.get ann CorefCoreAnnotations$CorefChainAnnotation)
          (.values)
          (mapv make-cluster))))

;;;;;;;;;;;;;;;;;;
(defonce penn-treebank-tags #{"CC"	; Coordinating conjunction
                              "CD"	; Cardinal number
                              "DT"	; Determiner
                              "EX"	; Existential there
                              "FW"	; Foreign word
                              "IN"	; Preposition or subordinating conjunction
                              "JJ"	; Adjective
                              "JJR"	; Adjective, comparative
                              "JJS"	; Adjective, superlative
                              "LS"	; List item marker
                              "MD"	; Modal
                              "NN"	; Noun, singular or mass
                              "NNS"	; Noun, plural
                              "NNP"	; Proper noun, singular
                              "NNPS"    ; Proper noun, plural
                              "PDT"	; Predeterminer
                              "POS"	; Possessive ending
                              "PRP"	; Personal pronoun
                              "PRP$"    ; Possessive pronoun
                              "RB"	; Adverb
                              "RBR"	; Adverb, comparative
                              "RBS"	; Adverb, superlative
                              "RP"	; Particle
                              "SYM"	; Symbol
                              "TO"	; to
                              "UH"	; Interjection
                              "VB"	; Verb, base form
                              "VBD"	; Verb, past tense
                              "VBG"	; Verb, gerund or present participle
                              "VBN"	; Verb, past participle
                              "VBP"	; Verb, non-3rd person singular present
                              "VBZ"	; Verb, 3rd person singular present
                              "WDT"	; Wh-determiner
                              "WP"	; Wh-pronoun
                              "WP$"	; Possessive wh-pronoun
                              "WRB"	; Wh-adverb
                              })

(defn penn-treebank-string->tag [s]
  (if-let [tag-str (penn-treebank-tags s)]
    (keyword tag-str)
    (throw (Exception. (str "Unknown peen-treebank tag: " s)))))

(defn str->word-and-penn-treebank-tag [s]
  (split s #"/"))

(defrecord WordDependency [governor-word governor-tag relation relation dependent-word dependent-tag])

(defn make-word-dependency [typed-dependency]
  (let [[governor-word governor-tag] (str->word-and-penn-treebank-tag
                                      (.toString (.gov typed-dependency)))
         [dependent-word dependent-tag] (str->word-and-penn-treebank-tag
                                         (.toString (.dep typed-dependency)))]
     (array-map :governor-word governor-word
                :governor-tag governor-tag
                :relation (-> (.reln typed-dependency) (.getLongName))
                :dependent-word dependent-word
                :dependent-tag dependent-tag)))


(defn- %load-model [model]
  (LexicalizedParser/loadModel (str "edu/stanford/nlp/models/lexparser/" model) []))

(defonce load-model (memoize %load-model))

(defn sentence->word-dependencies [sentence]
  (let [parser (load-model "englishPCFG.ser.gz")]
    (mapv make-word-dependency
          (-> (.treebankLanguagePack parser)
              (.grammaticalStructureFactory)
              (.newGrammaticalStructure (.parse parser (.tokenize (make-tokenizer :ptb-core-label sentence))))
              (.typedDependenciesCCprocessed)))))

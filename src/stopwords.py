"""Stop lists.

The paper used an extended version of the stop list from Jockers' *Macroanalysis*
(function words + ~5k personal names). Here: function words + spoken-dialogue
fillers; personal names are removed separately (pymorphy3 Name/Surn/Patr tags for
Russian, and a corpus-driven capitalisation heuristic for both languages, see prep.py).
"""
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

EN_EXTRA = """
i'm you're he's she's it's we're they're i've you've we've they've i'd you'd he'd she'd
we'd they'd i'll you'll he'll she'll it'll we'll they'll isn't aren't wasn't weren't
haven't hasn't hadn't don't doesn't didn't won't wouldn't can't cannot couldn't shouldn't
mustn't mightn't needn't let's that's there's here's what's who's where's when's why's
how's ain't y'all gonna wanna gotta gotcha lemme kinda sorta dunno ya yeah yes yep yup
nope no oh ah aah ahh uh uhh um umm hmm hm mm mmm huh hey hi okay ok o'k alright eh
er erm ha haha whoa wow ooh oops ow gonna doin goin nothin somethin 'em em just like
got get gets getting really right well know think want going go come thing things
said say says tell told mean little lot ll ve re don didn doesn isn wasn aren weren
couldn wouldn shouldn won hasn haven hadn mr mrs ms mister
""".split()

RU = """
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни
быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут
где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под
будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой
тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть
после над больше тот через эти нас про всего них какая много разве три эту моя впрочем
хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю
между весь это свой твой наш ваш который такой сам самый оно они тот мочь хотеть
сказать говорить знать ага угу ох ах эй ой хм эм ммм э а-а о-о да-да нет-нет ладно
окей ок типа вообще просто очень ещё уже вот-вот как-то что-то кто-то где-то когда-то
какой-то что-нибудь кто-нибудь ли-то либо же-то бы-то вон вроде значит то-есть итак
мистер миссис мисс
""".split()

STOP = {
    "EN": set(ENGLISH_STOP_WORDS) | set(EN_EXTRA),
    "RU": set(w.replace("ё", "е") for w in RU),
}

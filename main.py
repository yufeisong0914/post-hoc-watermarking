from models.methods.rsfph_wtgbblm import rsfph_wtgbblm

if __name__ == '__main__':
    wm_model = rsfph_wtgbblm.WatermarkModel()

    # step 1: embedding watermark
    text = "A game engine is a software platform that is designed to help game developers create video games more easily. It provides a set of tools and features that allow developers to create the various parts of a game, such as the graphics, sound, and gameplay mechanics, and then put all of these elements together in a way that allows the game to run on a computer or other device. Think of a game engine as a kind of \"toolbox\" that a game developer can use to build their game. It includes everything they need to create the game, from the basic building blocks (like graphics and sound) to more advanced features (like physics simulations and AI). By using a game engine, developers can focus on creating the fun and interesting parts of the game, rather than having to worry about all of the technical details. So, to sum it up: a game engine is a piece of software that helps game developers build and run video games."
    r1 = wm_model.watermark_text_generator(text)
    print(r1)
    ''' return
    {
        'watermarked_text': 'The game engine means an software platform that was designed to help game developers make video games very easily. It provides a set the tools plus features that allow developers to create the various parts of a game, such as the graphics, sound, and gameplay mechanics, and then put all of these elements together in a way that allows the game to run on a computer or other device. Think in a game engine like a kind like "toolbox" that a game developer might use to build their game. It includes everything they need to create the game, from the basic building blocks (like graphics and sound) into more advanced features (like physics simulations and AI). By using a game engine, developers might focus in creating the fun or interesting parts of the game, rather from having to worry about all of the technical details. So, to sum it over: the game engine means an piece in software that helps game developers build and run video games.', 
        'watermark_index': [
            [8, 0, 4, 3, 14, 17], 
            [6, 4], 
            [16, 1, 5, 8], 
            [21], 
            [13, 21, 7, 0, 9], 
            [13, 7, 11, 5, 10]
        ]
    }
    '''

    # step 2: detecting watermark
    # step 2.1: detecting watermarked text
    text_wm = r1['watermarked_text']
    r2 = wm_model.watermark_text_detector(text_wm)
    print(r2)
    ''' return
    {
        'watermarked': True, 
        'watermark_words': [
            [was, The, an, means, make, designed, very], 
            [plus, and, and, or, of, as, of, in, on, a, the, the], 
            [that, might, in, like, like], 
            [and, and, from, like, into, like, the, the, includes], 
            [or, might, By, in, of, from], 
            [and, in, the, an, over, sum, means]
        ], 
        'watermark_words_index': [
            [8, 0, 4, 3, 14, 9, 17], 
            [6, 27, 31, 51, 16, 21, 35, 39, 48, 2, 4, 13], 
            [12, 16, 1, 5, 8], 
            [18, 29, 10, 16, 21, 26, 7, 11, 1], 
            [13, 7, 0, 9, 16, 21], 
            [20, 13, 7, 11, 5, 3, 10]
        ], 
        'encoding': [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
        'ones/n': '41/46', 
        'z_score': 5.307910421576297, 
        'p_value': 5.5444555848024795e-08
    }
    '''
    # step 2.2: detecting original text
    r3 = wm_model.watermark_text_detector(text)
    print(r3)
    '''
    {
        'watermarked': False, 
        'watermark_words': [
            [is, A, a, is, create, designed, more], 
            [and, and, and, or, of, of, as, of, in, on, a, the], 
            [that, can, of, as, of], 
            [and, and, from, like, to, like, the, the, includes], 
            [and, than, can, By, on, of], 
            [and, of, a, a, up, sum, is]
        ], 
        'watermark_words_index': [
            [8, 0, 4, 3, 14, 9, 17], 
            [6, 27, 31, 51, 4, 16, 21, 35, 39, 48, 2, 13], 
            [12, 16, 1, 5, 8], 
            [18, 29, 10, 16, 21, 26, 7, 11, 1], 
            [13, 21, 7, 0, 9, 16], 
            [20, 13, 7, 11, 5, 3, 10]
        ], 
        'encoding': [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0], 
        'ones/n': '23/46', 
        'z_score': 0.0, 
        'p_value': 0.5
    }
    '''

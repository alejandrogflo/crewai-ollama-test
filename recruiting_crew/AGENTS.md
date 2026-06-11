## Pre-proceso de archivos

Antes de leer un archivo en formato no-texto (PDF), conviértelo primero a markdown con:

    markitdown <archivo> > <archivo>.md

Lee el `.md` resultante en lugar del original. Esto reduce el ruido (formato, encoding) y baja el consumo de contexto entre un 30% y un 60% según el archivo.
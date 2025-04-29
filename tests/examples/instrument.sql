-- DROP DATABASE IF EXISTS instrument WITH (FORCE);
CREATE DATABASE instrument WITH TEMPLATE template0 ENCODING = 'UTF8' LOCALE = 'en_US.utf8';
ALTER DATABASE instrument OWNER TO postgres;

\connect instrument

CREATE TABLE public.manufacturer (
    id INTEGER NOT NULL,
    name TEXT NOT NULL,
    founded TIMESTAMP WITH TIME ZONE NOT NULL
);

ALTER TABLE ONLY public.manufacturer ADD CONSTRAINT manufacturer_pkey PRIMARY KEY (id);

ALTER TABLE public.manufacturer OWNER TO postgres;

INSERT INTO public.manufacturer VALUES (1, 'Blender', 'January 8 04:05:06 1951 PST');
INSERT INTO public.manufacturer VALUES (2, 'Gibbs', 'March 4 07:08:09 1959 PST');

CREATE TABLE public.model (
    id INTEGER NOT NULL,
    name TEXT NOT NULL,
    manufacturer_id INTEGER NOT NULL,
    introduced TIMESTAMP WITH TIME ZONE NOT NULL
);

ALTER TABLE ONLY public.model ADD CONSTRAINT model_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.model
    ADD CONSTRAINT concept_manufacturer_id_fkey FOREIGN KEY (manufacturer_id) REFERENCES public.manufacturer(id);

ALTER TABLE public.model OWNER TO postgres;

INSERT INTO public.model VALUES (1, 'S-Type', 1, 'April 20 04:05:06 1952 PST');
INSERT INTO public.model VALUES (2, 'Pulse', 1, 'December 2 02:15:06 1953 PST');
INSERT INTO public.model VALUES (3, 'Paul Leslie', 2, 'February 20 04:05:06 1960 PST');

CREATE TABLE public.string (
    id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    frequency FLOAT NOT NULL
);

ALTER TABLE ONLY public.string ADD CONSTRAINT string_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.string
    ADD CONSTRAINT concept_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.model(id);

ALTER TABLE public.string OWNER TO postgres;

INSERT INTO public.string VALUES (1, 1, 1, 329.6);
INSERT INTO public.string VALUES (2, 1, 2, 246.94);
INSERT INTO public.string VALUES (3, 1, 3, 196);
INSERT INTO public.string VALUES (4, 1, 4, 146.83);
INSERT INTO public.string VALUES (5, 1, 5, 110);
INSERT INTO public.string VALUES (6, 1, 6, 82.4);
INSERT INTO public.string VALUES (7, 2, 1, 98);
INSERT INTO public.string VALUES (8, 2, 2, 73.42);
INSERT INTO public.string VALUES (9, 2, 3, 55);
INSERT INTO public.string VALUES (10, 2, 4, 30.87);
INSERT INTO public.string VALUES (11, 3, 1, 329.6);
INSERT INTO public.string VALUES (12, 3, 2, 246.94);
INSERT INTO public.string VALUES (13, 3, 3, 196);
INSERT INTO public.string VALUES (14, 3, 4, 146.83);
INSERT INTO public.string VALUES (15, 3, 5, 110);
INSERT INTO public.string VALUES (16, 3, 6, 82.4);
